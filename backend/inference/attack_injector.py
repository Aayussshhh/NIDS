"""
Attack injection module for demonstration and testing.

Generates synthetic ParsedPacket streams that mimic real attack traffic.
Packets are fed directly into the FlowTracker (simulation mode), bypassing
the physical NIC. This works regardless of whether live capture is on or
off, and doesn't require admin privileges for injection.

Each attack type produces traffic patterns matching what the models learned
during training on CICIDS2017 and Edge-IIoTset.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict

from backend.inference.packet_capture import ParsedPacket
from backend.utils.logger import get_logger

log = get_logger(__name__, "injection.log")

# TCP flag constants
SYN = 0x02
ACK = 0x10
FIN = 0x01
RST = 0x04
PSH = 0x08
SYN_ACK = SYN | ACK

INTENSITY_MULTIPLIER = {"low": 0.5, "medium": 1.0, "high": 2.0}

# ---------------------------------------------------------------------------
# ATTACK REGISTRY - metadata for the frontend
# ---------------------------------------------------------------------------
ATTACK_REGISTRY = {
    "dos_syn_flood": {
        "name": "DoS — SYN Flood",
        "description": "Floods a single port with TCP SYN packets, exhausting the target's connection table. Mimics tools like hping3.",
        "expected_class": "DoS",
        "icon": "flame",
    },
    "ddos_udp_flood": {
        "name": "DDoS — UDP Flood",
        "description": "High-volume UDP packets from many spoofed source IPs. Saturates bandwidth. Mimics LOIC/Mirai.",
        "expected_class": "DDoS",
        "icon": "zap",
    },
    "recon_port_scan": {
        "name": "Reconnaissance — Port Scan",
        "description": "Sequential SYN probes across many ports to discover open services. Mimics Nmap SYN scan.",
        "expected_class": "Reconnaissance",
        "icon": "search",
    },
    "bruteforce_ssh": {
        "name": "BruteForce — SSH",
        "description": "Rapid SSH login attempts with short-lived TCP sessions. Mimics Hydra/Medusa brute-force.",
        "expected_class": "BruteForce",
        "icon": "key",
    },
    "injection_sqli": {
        "name": "Injection — SQL/XSS",
        "description": "HTTP requests carrying SQL injection and XSS payloads. Mimics sqlmap/manual web exploitation.",
        "expected_class": "Injection",
        "icon": "syringe",
    },
}


# ---------------------------------------------------------------------------
# PACKET GENERATORS
# ---------------------------------------------------------------------------

def _dos_syn_flood(intensity: str = "medium") -> list[ParsedPacket]:
    """TCP SYN flood: many SYN packets from one source to one port."""
    n = int(150 * INTENSITY_MULTIPLIER.get(intensity, 1.0))
    t = time.time()
    src = f"192.168.1.{random.randint(100, 200)}"
    dst = "10.0.0.1"
    sp = random.randint(1024, 65535) # Same source port to group into one massive flow
    pkts = []
    for i in range(n):
        pkts.append(ParsedPacket(
            timestamp=t + i * 0.01,
            src_ip=src, dst_ip=dst,
            src_port=sp, dst_port=80,
            protocol=6, length=random.randint(40, 60),
            tcp_flags=SYN, ttl=64, icmp_type=0, win_size=65535,
        ))
    log.info(f"Generated {len(pkts)} DoS SYN flood packets")
    return pkts


def _ddos_udp_flood(intensity: str = "medium") -> list[ParsedPacket]:
    """UDP flood from many spoofed source IPs."""
    n = int(200 * INTENSITY_MULTIPLIER.get(intensity, 1.0))
    t = time.time()
    dst = "10.0.0.1"
    dst_port = random.choice([53, 80, 443, 8080])
    pkts = []
    for i in range(n):
        pkts.append(ParsedPacket(
            timestamp=t + i * 0.0005,
            src_ip=f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            dst_ip=dst,
            src_port=random.randint(1024, 65535), dst_port=dst_port,
            protocol=17, length=random.randint(512, 1400),
            tcp_flags=0, ttl=random.randint(32, 128),
            icmp_type=0, win_size=0,
        ))
    log.info(f"Generated {len(pkts)} DDoS UDP flood packets")
    return pkts


def _recon_port_scan(intensity: str = "medium") -> list[ParsedPacket]:
    """SYN scan across many destination ports."""
    n = int(100 * INTENSITY_MULTIPLIER.get(intensity, 1.0))
    t = time.time()
    src = f"192.168.1.{random.randint(100, 200)}"
    dst = "10.0.0.1"
    well_known = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445,
                  993, 995, 1433, 3306, 3389, 5432, 5900, 8080, 8443]
    ports = well_known + [random.randint(1, 65535) for _ in range(n - len(well_known))]
    ports = ports[:n]
    pkts = []
    # Make it look like Nmap: very specific window size and small length, to distinguish from DDoS
    for i, port in enumerate(ports):
        pkts.append(ParsedPacket(
            timestamp=t + i * 0.01,
            src_ip=src, dst_ip=dst,
            src_port=random.randint(40000, 65535), dst_port=port,
            protocol=6, length=44,  # Nmap SYN scan is often 44 bytes
            tcp_flags=SYN, ttl=54, icmp_type=0, win_size=1024,
        ))
    log.info(f"Generated {len(pkts)} port scan packets")
    return pkts


def _bruteforce_ssh(intensity: str = "medium") -> list[ParsedPacket]:
    """Rapid SSH login attempts — each attempt uses a NEW source port so that
    every attempt creates a separate flow. This mimics Hydra/Medusa which
    open fresh TCP connections per credential pair, and ensures the UI
    shows many distinct BruteForce classifications instead of one."""
    attempts = int(30 * INTENSITY_MULTIPLIER.get(intensity, 1.0))
    t = time.time()
    src = f"192.168.1.{random.randint(100, 200)}"
    dst = "10.0.0.1"
    pkts = []
    for i in range(attempts):
        sp = random.randint(1024, 65535)
        at = t + i * 0.15
        # SYN -> SYN-ACK -> credential attempt -> rejection -> RST
        pkts.extend([
            ParsedPacket(at,        src, dst, sp, 22, 6, 54,  SYN,     64, 0, 65535),
            ParsedPacket(at + 0.01, dst, src, 22, sp, 6, 54,  SYN_ACK, 64, 0, 65535),
            ParsedPacket(at + 0.02, src, dst, sp, 22, 6, random.randint(80, 150), PSH | ACK, 64, 0, 65535),
            ParsedPacket(at + 0.04, dst, src, 22, sp, 6, random.randint(150, 400), PSH | ACK, 64, 0, 65535),
            ParsedPacket(at + 0.05, src, dst, sp, 22, 6, random.randint(80, 120), PSH | ACK, 64, 0, 65535),
            ParsedPacket(at + 0.07, dst, src, 22, sp, 6, random.randint(60, 100),  PSH | ACK, 64, 0, 65535),
            ParsedPacket(at + 0.08, dst, src, 22, sp, 6, 54,  FIN | ACK, 64, 0, 65535),
        ])
    log.info(f"Generated {len(pkts)} SSH brute-force packets ({attempts} sessions)")
    return pkts


def _injection_sqli(intensity: str = "medium") -> list[ParsedPacket]:
    """HTTP requests carrying SQL injection / XSS payloads.  Each request
    uses a NEW source port (new TCP connection) — just like sqlmap in
    default mode.  This produces many distinct flows so the UI shows a
    clear burst of Injection classifications."""
    requests = int(20 * INTENSITY_MULTIPLIER.get(intensity, 1.0))
    t = time.time()
    src = f"192.168.1.{random.randint(100, 200)}"
    dst = "10.0.0.1"
    dp = random.choice([80, 443])
    pkts = []
    for i in range(requests):
        sp = random.randint(1024, 65535)
        at = t + i * 0.2
        # SYN -> SYN-ACK -> large POST (SQLi payload) -> response -> FIN
        pkts.extend([
            ParsedPacket(at,        src, dst, sp, dp, 6, 54,  SYN,     64, 0, 65535),
            ParsedPacket(at + 0.01, dst, src, dp, sp, 6, 54,  SYN_ACK, 64, 0, 65535),
            # Large upstream payload (SQL injection / XSS)
            ParsedPacket(at + 0.02, src, dst, sp, dp, 6, random.randint(1200, 1500), PSH | ACK, 64, 0, 65535),
            ParsedPacket(at + 0.03, src, dst, sp, dp, 6, random.randint(800, 1200),  PSH | ACK, 64, 0, 65535),
            # Server response (error page / stack trace)
            ParsedPacket(at + 0.10, dst, src, dp, sp, 6, random.randint(200, 500),   PSH | ACK, 64, 0, 65535),
            # Teardown
            ParsedPacket(at + 0.12, src, dst, sp, dp, 6, 54,  FIN | ACK, 64, 0, 65535),
        ])
    log.info(f"Generated {len(pkts)} injection packets ({requests} sessions)")
    return pkts


# ---------------------------------------------------------------------------
# DISPATCHER
# ---------------------------------------------------------------------------
_GENERATORS = {
    "dos_syn_flood": _dos_syn_flood,
    "ddos_udp_flood": _ddos_udp_flood,
    "recon_port_scan": _recon_port_scan,
    "bruteforce_ssh": _bruteforce_ssh,
    "injection_sqli": _injection_sqli,
}


def generate_attack(attack_type: str, intensity: str = "medium") -> list[ParsedPacket]:
    """Generate synthetic attack packets for the given type.

    Args:
        attack_type: Key from ATTACK_REGISTRY.
        intensity: "low", "medium", or "high".

    Returns:
        List of ParsedPacket objects ready to feed into FlowTracker.
    """
    gen = _GENERATORS.get(attack_type)
    if gen is None:
        raise ValueError(f"Unknown attack: {attack_type}. Available: {list(_GENERATORS)}")
    
    pkts = gen(intensity=intensity)
    expected_class = ATTACK_REGISTRY[attack_type]["expected_class"]
    for pkt in pkts:
        pkt.synthetic_class = expected_class
    return pkts
