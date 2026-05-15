"""
Real-time flow tracker / feature extractor.

Aggregates the stream of ParsedPackets from packet_capture.py into
"flows" (5-tuple-keyed bidirectional sessions) and emits them as
feature vectors aligned with the NF-v2 schema once they expire.

A flow expires when:
    - It has been idle for FLOW_INACTIVE_TIMEOUT_SECONDS (default 15s),
      OR
    - It has been alive for FLOW_TIMEOUT_SECONDS (default 60s),
      OR
    - It sees a TCP FIN or RST.

Key design: this module never blocks on the classifier. Expired flows
are pushed into an asyncio.Queue for the predictor to consume at its
own pace. If the predictor falls behind, flows are dropped and counted
- preferable to OOM-ing the process.

Limitations vs. a full NetFlow exporter (nProbe, fprobe):
    - We don't compute every NF-v2 feature exactly the same way nProbe
      does (we lack DNS/FTP application-layer parsing). The features we
      can't compute online are filled with 0; the model already sees
      this pattern from the Edge-IIoTset side of training, so it's
      robust to the gaps.
    - For an industry-grade deployment you would replace this with
      nProbe + a Kafka topic and read flows from there. For the live
      demo, this in-process version is far simpler and adequate.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.config import (
    CATEGORICAL_FEATURES,
    FLOW_INACTIVE_TIMEOUT_SECONDS,
    FLOW_TIMEOUT_SECONDS,
    MAX_TRACKED_FLOWS,
    NF_V2_FEATURES,
    SCALERS_DIR,
    TRAINING_FEATURES,
)
from backend.inference.packet_capture import ParsedPacket
from backend.utils.logger import get_logger
import joblib

log = get_logger(__name__, "inference.log")


# TCP flag bit positions (matches the byte layout in TCP headers).
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20


@dataclass
class FlowState:
    """Mutable per-flow state. We keep this minimal: just enough to
    compute every NF-v2 feature when the flow expires.

    The "client" direction is the side that sent the SYN (or the first
    packet, for non-TCP). Everything else is the "server" direction.
    """
    # 5-tuple
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int

    # Timing
    first_seen: float
    last_seen: float

    # Per-direction packet/byte counts
    in_pkts: int = 0     # client -> server
    in_bytes: int = 0
    out_pkts: int = 0    # server -> client
    out_bytes: int = 0

    # TCP flag accumulators (OR-ed across packets)
    client_tcp_flags: int = 0
    server_tcp_flags: int = 0

    # Packet length stats
    pkt_lens: list[int] = field(default_factory=list)

    # TTL stats
    ttls: list[int] = field(default_factory=list)

    # TCP window
    tcp_win_max_in: int = 0
    tcp_win_max_out: int = 0

    # ICMP
    icmp_type: int = 0

    # Number-of-packets buckets (NF-v2 has these)
    pkts_up_to_128: int = 0
    pkts_128_256: int = 0
    pkts_256_512: int = 0
    pkts_512_1024: int = 0
    pkts_1024_1514: int = 0
    
    synthetic_class: str | None = None


def _flow_key(pkt: ParsedPacket) -> tuple:
    """Bidirectional 5-tuple key.

    Sort the (ip,port) endpoints so that A->B and B->A map to the same
    key. This is how Wireshark, NetFlow, and CICFlowMeter all do it.
    """
    a = (pkt.src_ip, pkt.src_port)
    b = (pkt.dst_ip, pkt.dst_port)
    if a < b:
        return (a, b, pkt.protocol)
    return (b, a, pkt.protocol)


def _is_client_to_server(state: FlowState, pkt: ParsedPacket) -> bool:
    """Determine packet direction relative to the flow's stored 'client'."""
    return pkt.src_ip == state.src_ip and pkt.src_port == state.src_port


@dataclass
class CompletedFlow:
    """A flow that has expired and is ready for classification.

    Includes both the feature vector (for the model) and the raw 5-tuple
    + timing (for display in the UI / logs).
    """
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    first_seen: float
    last_seen: float
    feature_vector: np.ndarray  # ordered as TRAINING_FEATURES
    synthetic_class: str | None = None


class FlowTracker:
    """Maintains active flows and emits CompletedFlow objects on expiration.

    Reuses the same RobustScaler + LabelEncoders that were fit during
    training, so live features are scaled identically to training data.
    """

    def __init__(self, output_queue: asyncio.Queue):
        self._flows: dict[tuple, FlowState] = {}
        self._output_queue = output_queue
        self._sweep_task: asyncio.Task | None = None
        # Reload preprocessing artifacts saved during training.
        self._scaler = joblib.load(SCALERS_DIR / "robust_scaler.joblib")
        if hasattr(self._scaler, "feature_names_in_"):
            delattr(self._scaler, "feature_names_in_")
        self._encoders = {
            col: joblib.load(SCALERS_DIR.parent / "encoders" / f"{col}_encoder.joblib")
            for col in CATEGORICAL_FEATURES
        }
        # Stats
        self.flows_emitted = 0
        self.flows_dropped = 0
        self.active_count = 0

    def add_packet(self, pkt: ParsedPacket) -> None:
        """Update the flow state for this packet's 5-tuple."""
        # Cap memory: if we're at the limit, evict the oldest flow.
        if len(self._flows) >= MAX_TRACKED_FLOWS:
            self._evict_oldest()

        key = _flow_key(pkt)
        state = self._flows.get(key)
        if state is None:
            state = FlowState(
                src_ip=pkt.src_ip,
                dst_ip=pkt.dst_ip,
                src_port=pkt.src_port,
                dst_port=pkt.dst_port,
                protocol=pkt.protocol,
                first_seen=pkt.timestamp,
                last_seen=pkt.timestamp,
            )
            self._flows[key] = state

        state.last_seen = pkt.timestamp
        state.pkt_lens.append(pkt.length)
        state.ttls.append(pkt.ttl)
        state.icmp_type = pkt.icmp_type or state.icmp_type
        if pkt.synthetic_class:
            state.synthetic_class = pkt.synthetic_class

        if _is_client_to_server(state, pkt):
            state.in_pkts += 1
            state.in_bytes += pkt.length
            state.client_tcp_flags |= pkt.tcp_flags
            state.tcp_win_max_in = max(state.tcp_win_max_in, pkt.win_size)
        else:
            state.out_pkts += 1
            state.out_bytes += pkt.length
            state.server_tcp_flags |= pkt.tcp_flags
            state.tcp_win_max_out = max(state.tcp_win_max_out, pkt.win_size)

        # Update packet-size histogram (NF-v2 specifies these bins).
        L = pkt.length
        if L <= 128: state.pkts_up_to_128 += 1
        elif L <= 256: state.pkts_128_256 += 1
        elif L <= 512: state.pkts_256_512 += 1
        elif L <= 1024: state.pkts_512_1024 += 1
        else: state.pkts_1024_1514 += 1

        # Immediate expiration on FIN/RST.
        if pkt.tcp_flags & (TCP_FIN | TCP_RST):
            self._emit(key)
        self.active_count = len(self._flows)

    def _evict_oldest(self) -> None:
        """Remove the flow with the earliest first_seen. Counts as dropped."""
        oldest_key = min(self._flows, key=lambda k: self._flows[k].first_seen)
        self._flows.pop(oldest_key, None)
        self.flows_dropped += 1

    def _emit(self, key: tuple) -> None:
        """Build a CompletedFlow from a flow state and enqueue it."""
        state = self._flows.pop(key, None)
        if state is None:
            return

        completed = self._build_completed(state)
        try:
            self._output_queue.put_nowait(completed)
            self.flows_emitted += 1
        except asyncio.QueueFull:
            self.flows_dropped += 1

    def _build_completed(self, state: FlowState) -> CompletedFlow:
        """Compute every NF-v2 feature from the accumulated state."""
        duration_ms = max(0.001, (state.last_seen - state.first_seen) * 1000.0)
        duration_s = duration_ms / 1000.0

        pkt_lens = state.pkt_lens or [0]
        ttls = state.ttls or [64]

        # Build a dict of NF-v2 feature name -> value.
        nfv2 = {f: 0.0 for f in NF_V2_FEATURES}
        nfv2["L4_SRC_PORT"] = state.src_port
        nfv2["L4_DST_PORT"] = state.dst_port
        nfv2["PROTOCOL"] = state.protocol
        nfv2["L7_PROTO"] = 0
        nfv2["IN_BYTES"] = state.in_bytes
        nfv2["IN_PKTS"] = state.in_pkts
        nfv2["OUT_BYTES"] = state.out_bytes
        nfv2["OUT_PKTS"] = state.out_pkts
        nfv2["TCP_FLAGS"] = state.client_tcp_flags | state.server_tcp_flags
        nfv2["CLIENT_TCP_FLAGS"] = state.client_tcp_flags
        nfv2["SERVER_TCP_FLAGS"] = state.server_tcp_flags
        nfv2["FLOW_DURATION_MILLISECONDS"] = duration_ms
        nfv2["DURATION_IN"] = duration_s
        nfv2["DURATION_OUT"] = duration_s
        nfv2["MIN_TTL"] = min(ttls)
        nfv2["MAX_TTL"] = max(ttls)
        nfv2["LONGEST_FLOW_PKT"] = max(pkt_lens)
        nfv2["SHORTEST_FLOW_PKT"] = min(pkt_lens)
        nfv2["MIN_IP_PKT_LEN"] = min(pkt_lens)
        nfv2["MAX_IP_PKT_LEN"] = max(pkt_lens)
        nfv2["SRC_TO_DST_SECOND_BYTES"] = state.in_bytes / max(duration_s, 0.001)
        nfv2["DST_TO_SRC_SECOND_BYTES"] = state.out_bytes / max(duration_s, 0.001)
        nfv2["SRC_TO_DST_AVG_THROUGHPUT"] = nfv2["SRC_TO_DST_SECOND_BYTES"] * 8
        nfv2["DST_TO_SRC_AVG_THROUGHPUT"] = nfv2["DST_TO_SRC_SECOND_BYTES"] * 8
        nfv2["NUM_PKTS_UP_TO_128_BYTES"] = state.pkts_up_to_128
        nfv2["NUM_PKTS_128_TO_256_BYTES"] = state.pkts_128_256
        nfv2["NUM_PKTS_256_TO_512_BYTES"] = state.pkts_256_512
        nfv2["NUM_PKTS_512_TO_1024_BYTES"] = state.pkts_512_1024
        nfv2["NUM_PKTS_1024_TO_1514_BYTES"] = state.pkts_1024_1514
        nfv2["TCP_WIN_MAX_IN"] = state.tcp_win_max_in
        nfv2["TCP_WIN_MAX_OUT"] = state.tcp_win_max_out
        nfv2["ICMP_TYPE"] = state.icmp_type
        nfv2["ICMP_IPV4_TYPE"] = state.icmp_type

        # Apply training-time preprocessing: encode categoricals,
        # log1p + RobustScaler the numerics. ORDER MUST MATCH TRAINING.
        feature_vector = self._preprocess(nfv2)

        return CompletedFlow(
            src_ip=state.src_ip,
            dst_ip=state.dst_ip,
            src_port=state.src_port,
            dst_port=state.dst_port,
            protocol=state.protocol,
            first_seen=state.first_seen,
            last_seen=state.last_seen,
            feature_vector=feature_vector,
            synthetic_class=state.synthetic_class,
        )

    def _preprocess(self, nfv2: dict[str, float]) -> np.ndarray:
        """Apply identical preprocessing as training (encoders + scaler)."""
        # Encode categoricals.
        for col in CATEGORICAL_FEATURES:
            le = self._encoders[col]
            val = str(int(nfv2[col]))
            known = set(le.classes_)
            nfv2[col] = float(
                le.transform([val])[0] if val in known else len(le.classes_)
            )

        # Build vector in TRAINING_FEATURES order.
        vec = np.array([nfv2[c] for c in TRAINING_FEATURES], dtype=np.float32)

        # Numeric: log1p + RobustScaler. The scaler was fit on log1p'd
        # data, so we reproduce that here.
        numeric_idx = [i for i, c in enumerate(TRAINING_FEATURES)
                       if c not in CATEGORICAL_FEATURES]
        vec[numeric_idx] = np.log1p(np.clip(vec[numeric_idx], 0, None))
        vec_2d = vec.reshape(1, -1).copy()
        vec_2d[:, numeric_idx] = self._scaler.transform(vec_2d[:, numeric_idx])
        return vec_2d.flatten()

    async def sweep_loop(self, interval: float = 1.0) -> None:
        """Background coroutine: every `interval` seconds, expire stale flows.

        Run alongside the packet ingest loop. Cancel to stop.
        """
        log.info(f"FlowTracker sweep loop started (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)
                now = time.time()
                expired_keys = [
                    key for key, st in self._flows.items()
                    if (now - st.last_seen) > FLOW_INACTIVE_TIMEOUT_SECONDS
                    or (now - st.first_seen) > FLOW_TIMEOUT_SECONDS
                ]
                for key in expired_keys:
                    self._emit(key)
                self.active_count = len(self._flows)
            except asyncio.CancelledError:
                log.info("FlowTracker sweep loop cancelled")
                break
            except Exception as e:
                log.exception(f"sweep_loop error: {e}")

    def get_stats(self) -> dict:
        return {
            "active_flows": self.active_count,
            "emitted": self.flows_emitted,
            "dropped": self.flows_dropped,
        }
