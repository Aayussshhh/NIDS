"""
Live network packet capture.

Wraps Scapy's AsyncSniffer in a thread-safe class that emits parsed
packet records into an asyncio.Queue. The queue is consumed by the
flow tracker (feature_extractor.py), which assembles packets into
flows for classification.

Windows requirements:
    - Npcap installed (https://npcap.com). Scapy uses it transparently.
    - The Python process needs to run with administrator privileges to
      open a packet sniffer. The README documents this.

Why Scapy + Npcap (and not pyshark or pcapy):
    - Scapy is pure Python with optional Npcap acceleration; cleanest
      install on Windows.
    - pyshark requires a separate tshark binary install.
    - pcapy needs Visual C++ build tools.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

from scapy.all import AsyncSniffer, Packet
from scapy.layers.inet import IP, TCP, UDP, ICMP

from backend.utils.logger import get_logger

log = get_logger(__name__, "inference.log")


@dataclass
class ParsedPacket:
    """A minimal, JSON-serializable view of one captured packet.

    We extract only the fields we need for flow construction, both to
    keep memory low on busy networks and to avoid the (slow) Scapy
    show()/dict() machinery.
    """
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int       # IP protocol number: 6=TCP, 17=UDP, 1=ICMP
    length: int         # IP packet length in bytes
    tcp_flags: int      # 0 if not TCP
    ttl: int
    icmp_type: int      # 0 if not ICMP
    win_size: int       # TCP window size, 0 if not TCP
    synthetic_class: str | None = None  # If injected by demo, forces classification


def _parse(pkt: Packet) -> ParsedPacket | None:
    """Convert a Scapy Packet to ParsedPacket. Returns None for non-IP."""
    if IP not in pkt:
        # Drop ARP, IPv6, and non-routable layer-2 frames. Production
        # NIDS would handle IPv6 separately; out of scope here.
        return None

    ip = pkt[IP]
    src_port = dst_port = 0
    tcp_flags = win_size = 0
    icmp_type = 0

    if TCP in pkt:
        tcp = pkt[TCP]
        src_port = int(tcp.sport)
        dst_port = int(tcp.dport)
        tcp_flags = int(tcp.flags)
        win_size = int(tcp.window)
    elif UDP in pkt:
        udp = pkt[UDP]
        src_port = int(udp.sport)
        dst_port = int(udp.dport)
    elif ICMP in pkt:
        icmp_type = int(pkt[ICMP].type)

    return ParsedPacket(
        timestamp=float(pkt.time),
        src_ip=str(ip.src),
        dst_ip=str(ip.dst),
        src_port=src_port,
        dst_port=dst_port,
        protocol=int(ip.proto),
        length=int(ip.len),
        tcp_flags=tcp_flags,
        ttl=int(ip.ttl),
        icmp_type=icmp_type,
        win_size=win_size,
    )


class PacketCapture:
    """Thread-based packet sniffer with an asyncio-compatible output queue.

    Usage:
        capture = PacketCapture(interface="Ethernet")
        capture.start()
        while True:
            pkt = await capture.queue.get()
            ...
        capture.stop()
    """

    def __init__(
        self,
        interface: str | None = None,
        bpf_filter: str | None = "ip",
        queue_max: int = 10_000,
    ):
        """
        Args:
            interface: Network interface name. None means Scapy's default.
                On Windows, use names from `scripts/list_interfaces.py`.
            bpf_filter: Berkeley Packet Filter, default "ip" for IPv4.
                Use this to focus on specific traffic
                (e.g. "tcp port 80" for HTTP only).
            queue_max: Max buffered packets. If full, oldest are dropped.
                Set high for noisy networks; low to fail fast under
                overload (which surfaces backpressure to the operator).
        """
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.queue: asyncio.Queue[ParsedPacket] = asyncio.Queue(maxsize=queue_max)
        self._sniffer: AsyncSniffer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._stats = {"captured": 0, "dropped": 0, "started_at": 0.0}

    def _on_packet(self, pkt: Packet) -> None:
        """Scapy callback - runs on the sniffer thread."""
        parsed = _parse(pkt)
        if parsed is None:
            return
        with self._lock:
            self._stats["captured"] += 1
        # Cross-thread enqueue to the asyncio loop. call_soon_threadsafe
        # is the safe way to interact with an event loop from a
        # background thread.
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue_nowait, parsed)
        except RuntimeError:
            # Loop closed mid-callback; ignore.
            pass

    def _enqueue_nowait(self, item: ParsedPacket) -> None:
        """Try to enqueue without blocking. Drop on overflow."""
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            with self._lock:
                self._stats["dropped"] += 1

    def start(self) -> None:
        """Start capturing. Captures the current asyncio loop for thread-
        safe queue interaction; must be called from inside an async
        function (or after asyncio.run() has set up a loop)."""
        if self._sniffer is not None:
            log.warning("PacketCapture already running.")
            return
        self._loop = asyncio.get_event_loop()
        self._stats["started_at"] = time.time()
        self._sniffer = AsyncSniffer(
            iface=self.interface,
            filter=self.bpf_filter,
            prn=self._on_packet,
            store=False,  # don't keep packets in memory after callback
        )
        self._sniffer.start()
        log.info(f"PacketCapture started on iface={self.interface}, "
                 f"filter={self.bpf_filter}")

    def stop(self) -> None:
        if self._sniffer is None:
            return
        self._sniffer.stop()
        self._sniffer = None
        log.info(f"PacketCapture stopped. Stats: {self.get_stats()}")

    def get_stats(self) -> dict:
        with self._lock:
            stats = dict(self._stats)
        if stats["started_at"]:
            elapsed = time.time() - stats["started_at"]
            stats["pps"] = stats["captured"] / elapsed if elapsed > 0 else 0.0
        return stats


class SyntheticCapture:
    """A drop-in replacement for PacketCapture that generates benign traffic
    instead of sniffing a real network interface.
    """

    def __init__(self, interface: str | None = None, bpf_filter: str | None = "ip", queue_max: int = 10_000):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.queue: asyncio.Queue[ParsedPacket] = asyncio.Queue(maxsize=queue_max)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._lock = threading.Lock()
        self._stats = {"captured": 0, "dropped": 0, "started_at": 0.0}
        self.is_running = False
        self.is_paused = False

    async def _generator_loop(self):
        import random
        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue
            try:
                # Generate 1 to 5 packets every 0.1s (~30 pps)
                count = random.randint(1, 5)
                for _ in range(count):
                    # Simulate HTTP traffic
                    pkt = ParsedPacket(
                        timestamp=time.time(),
                        src_ip=f"192.168.1.{random.randint(10, 250)}",
                        dst_ip=f"10.0.0.{random.randint(1, 100)}",
                        src_port=random.randint(1024, 65535),
                        dst_port=random.choice([80, 443, 8080]),
                        protocol=6, # TCP
                        length=random.randint(64, 1500),
                        tcp_flags=24, # PSH | ACK
                        ttl=64,
                        icmp_type=0,
                        win_size=65535,
                        synthetic_class="Benign"
                    )
                    
                    with self._lock:
                        self._stats["captured"] += 1
                        
                    try:
                        self.queue.put_nowait(pkt)
                    except asyncio.QueueFull:
                        with self._lock:
                            self._stats["dropped"] += 1
                            
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception(f"SyntheticCapture error: {e}")
                await asyncio.sleep(1)

    def start(self) -> None:
        if self.is_running:
            return
        self._loop = asyncio.get_event_loop()
        self.is_running = True
        self._stats["started_at"] = time.time()
        self._task = self._loop.create_task(self._generator_loop(), name="synthetic_capture")
        log.info(f"SyntheticCapture started")

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        if self._task:
            self._task.cancel()
            self._task = None
        log.info(f"SyntheticCapture stopped. Stats: {self.get_stats()}")

    def get_stats(self) -> dict:
        with self._lock:
            stats = dict(self._stats)
        if stats["started_at"]:
            elapsed = time.time() - stats["started_at"]
            stats["pps"] = stats["captured"] / elapsed if elapsed > 0 else 0.0
        return stats
