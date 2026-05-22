"""
FastAPI server.

Exposes:
    GET  /                      health check
    GET  /api/interfaces        list available network interfaces
    GET  /api/stats             pipeline + model stats
    GET  /api/recent            recent classifications (last 100)
    GET  /api/attack-types      list injectable attack types
    POST /api/start             start live capture+classification
    POST /api/stop              stop everything
    POST /api/inject            inject synthetic attack traffic
    POST /api/capture-toggle    enable/disable real-time capture
    POST /api/classify          classify a single uploaded flow (REST)
    WS   /ws/live               WebSocket stream of classifications
    WS   /ws/stats              WebSocket stream of pipeline stats

Lifecycle:
    The pipeline (capture -> tracker -> predictor -> result queue) runs
    as background asyncio tasks managed by the lifespan context. Hitting
    /api/start spins them up; /api/stop tears them down. The WebSocket
    handlers only stream what's already in the result queue - they don't
    affect the pipeline rate.

Modularity:
    Real-time capture can be toggled independently of the prediction
    pipeline. When capture is off, the tracker + predictor remain active
    and the attack injector can still feed synthetic traffic through them.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import (
    ALLOWED_ORIGINS,
    API_HOST,
    API_PORT,
    DEFAULT_INTERFACE,
    UNIFIED_LABELS,
    WEBSOCKET_BROADCAST_HZ,
)
from backend.inference.attack_injector import ATTACK_REGISTRY, generate_attack
from backend.inference.feature_extractor import FlowTracker
from backend.inference.packet_capture import PacketCapture, SyntheticCapture
from backend.inference.predictor import HybridPredictor
from backend.utils.logger import get_logger

log = get_logger(__name__, "api.log")

# ---------------------------------------------------------------------------
# GLOBAL APP STATE
# ---------------------------------------------------------------------------
class PipelineState:
    def __init__(self):
        self.predictor: HybridPredictor | None = None
        self.capture: SyntheticCapture | None = None
        self.tracker: FlowTracker | None = None
        self.flow_queue: asyncio.Queue | None = None
        self.result_queue: asyncio.Queue | None = None
        self.tasks: list[asyncio.Task] = []
        self.recent: deque[dict] = deque(maxlen=2000)
        self.is_running = False
        self.capture_enabled = True  # Can be toggled independently
        self.current_interface: str | None = None
        self.current_bpf: str | None = None
        self.total_processed: int = 0


state = PipelineState()


# ---------------------------------------------------------------------------
# LIFECYCLE
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models at startup so the first /api/start is fast.
    Stop everything on shutdown."""
    log.info("Loading models on startup...")
    try:
        state.predictor = HybridPredictor(enable_explainability=True)
    except Exception as e:
        log.error(f"Could not load models on startup: {e}")
        log.error("API will start but /api/start will fail until models are trained.")
    yield
    await stop_pipeline()
    log.info("API shutdown complete.")


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hybrid NIDS",
    description="XGBoost + CNN-BiLSTM-Attention + Conv-AE + SHAP",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    interface: str | None = None
    bpf_filter: str | None = "ip"
    capture_enabled: bool = True


class InjectRequest(BaseModel):
    attack_type: str
    intensity: str = "medium"


class CaptureToggleRequest(BaseModel):
    enabled: bool


class StatsResponse(BaseModel):
    is_running: bool
    capture_enabled: bool
    capture: dict | None
    tracker: dict | None
    predictor: dict | None
    classes: list[str]


# ---------------------------------------------------------------------------
# PIPELINE CONTROL
# ---------------------------------------------------------------------------
async def start_pipeline(
    interface: str | None, bpf: str | None, capture_enabled: bool = True
) -> None:
    """Build the async pipeline: [capture ->] tracker -> predictor."""
    if state.is_running:
        raise HTTPException(400, "Pipeline already running. Stop it first.")
    if state.predictor is None:
        raise HTTPException(503, "Models not loaded. Train models first.")

    state.flow_queue = asyncio.Queue(maxsize=10_000)
    state.result_queue = asyncio.Queue(maxsize=1_000)
    state.capture_enabled = capture_enabled
    state.current_interface = interface
    state.current_bpf = bpf

    # Tracker always runs (needed for both capture and injection).
    state.tracker = FlowTracker(state.flow_queue)

    tasks = []

    # Capture (optional - can be toggled).
    if capture_enabled:
        _start_capture(interface, bpf, tasks)

    # Glue task: drain result queue -> recent buffer.
    async def result_to_recent():
        try:
            while True:
                result = await state.result_queue.get()
                state.recent.append(result)
                state.total_processed += 1
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.exception(f"result_to_recent error: {e}")

    tasks.extend([
        asyncio.create_task(state.tracker.sweep_loop(), name="flow_sweeper"),
        asyncio.create_task(
            state.predictor.consume_loop(state.flow_queue, state.result_queue),
            name="predictor",
        ),
        asyncio.create_task(result_to_recent(), name="result_to_recent"),
    ])

    state.tasks = tasks
    state.is_running = True
    log.info(f"Pipeline started (iface={interface}, capture={capture_enabled})")


def _start_capture(interface, bpf, tasks):
    """Start packet capture and wire it to the tracker."""
    state.capture = SyntheticCapture(
        interface=interface or DEFAULT_INTERFACE, bpf_filter=bpf
    )
    state.capture.start()

    async def packet_to_tracker():
        try:
            while True:
                pkt = await state.capture.queue.get()
                state.tracker.add_packet(pkt)
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.exception(f"packet_to_tracker error: {e}")

    tasks.append(
        asyncio.create_task(packet_to_tracker(), name="packet_to_tracker")
    )


async def _stop_capture() -> None:
    """Stop only the capture component, leaving tracker+predictor running."""
    if state.capture is not None:
        state.capture.stop()
        state.capture = None
    # Cancel only the packet_to_tracker task.
    for t in list(state.tasks):
        if t.get_name() == "packet_to_tracker":
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            state.tasks.remove(t)


async def _resume_capture(interface, bpf) -> None:
    """Resume capture after it was stopped."""
    new_tasks = []
    _start_capture(interface, bpf, new_tasks)
    state.tasks.extend(new_tasks)


async def stop_pipeline() -> None:
    """Cancel all background tasks and stop the sniffer."""
    if not state.is_running:
        return
    log.info("Stopping pipeline...")
    if state.capture is not None:
        state.capture.stop()
    for t in state.tasks:
        t.cancel()
    await asyncio.gather(*state.tasks, return_exceptions=True)
    state.tasks = []
    state.capture = None
    state.tracker = None
    state.flow_queue = None
    state.result_queue = None
    state.is_running = False
    state.capture_enabled = True


# ---------------------------------------------------------------------------
# REST ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "Hybrid NIDS",
        "status": "ok",
        "models_loaded": state.predictor is not None,
        "is_running": state.is_running,
    }


@app.get("/api/interfaces")
async def list_interfaces():
    """List network interfaces visible to Scapy."""
    try:
        from scapy.arch import get_if_list, get_if_addr  # noqa: WPS433
        ifs = []
        for name in get_if_list():
            try:
                addr = get_if_addr(name)
            except Exception:
                addr = ""
            ifs.append({"name": name, "address": addr})
        return {"interfaces": ifs}
    except Exception as e:
        log.exception("Failed to list interfaces")
        raise HTTPException(500, f"Could not enumerate interfaces: {e}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    return StatsResponse(
        is_running=state.is_running,
        capture_enabled=state.capture_enabled,
        capture=state.capture.get_stats() if state.capture else None,
        tracker=state.tracker.get_stats() if state.tracker else None,
        predictor=state.predictor.get_stats() if state.predictor else None,
        classes=UNIFIED_LABELS,
    )


@app.get("/api/recent")
async def get_recent(limit: int = 100):
    """Return the most recent classifications. Newest first."""
    items = list(state.recent)[-limit:]
    items.reverse()
    return {"results": items, "count": len(items)}


@app.get("/api/attack-types")
async def get_attack_types():
    """Return the registry of injectable attack types."""
    return {"attack_types": ATTACK_REGISTRY}


@app.post("/api/start")
async def start(req: StartRequest):
    await start_pipeline(req.interface, req.bpf_filter, req.capture_enabled)
    return {
        "status": "started",
        "interface": req.interface,
        "capture_enabled": req.capture_enabled,
    }


@app.post("/api/stop")
async def stop():
    await stop_pipeline()
    return {"status": "stopped"}


@app.post("/api/capture-toggle")
async def capture_toggle(req: CaptureToggleRequest):
    """Enable or disable real-time packet capture without stopping the
    prediction pipeline. Injection still works either way."""
    if not state.is_running:
        raise HTTPException(400, "Pipeline not running. Start it first.")

    if req.enabled and not state.capture_enabled:
        # Resume capture.
        await _resume_capture(
            state.current_interface, state.current_bpf
        )
        state.capture_enabled = True
        log.info("Real-time capture RESUMED")
    elif not req.enabled and state.capture_enabled:
        # Pause capture.
        await _stop_capture()
        state.capture_enabled = False
        log.info("Real-time capture PAUSED")

    return {"capture_enabled": state.capture_enabled}


@app.post("/api/inject")
async def inject(req: InjectRequest):
    """Inject synthetic attack traffic into the prediction pipeline.

    Works regardless of whether real-time capture is on or off. Packets
    are fed directly into the FlowTracker as ParsedPacket objects.

    To prevent attack results from being buried under the continuous
    benign traffic stream, we temporarily pause the synthetic capture
    during injection and give the predictor time to process the attack
    flows before resuming.
    """
    if not state.is_running:
        raise HTTPException(400, "Pipeline not running. Start it first.")
    if state.tracker is None:
        raise HTTPException(500, "FlowTracker not initialized.")

    try:
        packets = generate_attack(req.attack_type, req.intensity)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Temporarily pause synthetic capture so attack flows aren't drowned.
    capture_was_running = False
    if state.capture is not None and getattr(state.capture, 'is_running', False):
        capture_was_running = True
        if hasattr(state.capture, 'is_paused'):
            state.capture.is_paused = True
        # Drain the capture queue so stale benign packets don't mix in.
        while not state.capture.queue.empty():
            try:
                state.capture.queue.get_nowait()
            except Exception:
                break

    # Force a sweep BEFORE injecting, so pending benign flows are emitted first
    # and appear older (lower in the UI) than the new attack flows.
    expired_keys = list(state.tracker._flows.keys())
    for key in expired_keys:
        state.tracker._emit(key)

    # Feed packets into the tracker.
    for pkt in packets:
        state.tracker.add_packet(pkt)

    # Force another sweep to emit the attack flows immediately (for those without FIN).
    expired_keys = list(state.tracker._flows.keys())
    for key in expired_keys:
        state.tracker._emit(key)

    # Give the predictor consume_loop time to process the attack flows.
    # Without this, the endpoint returns before results reach the recent
    # buffer, and the WebSocket may not broadcast them before benign
    # traffic resumes and buries them.
    await asyncio.sleep(0.5)

    # Resume synthetic capture.
    if capture_was_running:
        if hasattr(state.capture, 'is_paused'):
            state.capture.is_paused = False

    attack_info = ATTACK_REGISTRY.get(req.attack_type, {})
    log.info(f"Injected {len(packets)} packets for {req.attack_type} "
             f"(intensity={req.intensity})")

    return {
        "status": "injected",
        "attack_type": req.attack_type,
        "packets_generated": len(packets),
        "expected_class": attack_info.get("expected_class", "Unknown"),
        "intensity": req.intensity,
    }


@app.post("/api/explain/{flow_index}")
async def explain(flow_index: int):
    """On-demand SHAP explanation for a flow already in /api/recent."""
    if state.predictor is None:
        raise HTTPException(503, "Models not loaded.")
    items = list(state.recent)
    if not items or flow_index >= len(items):
        raise HTTPException(404, f"Flow index {flow_index} not in recent buffer")
    return items[flow_index]


# ---------------------------------------------------------------------------
# WEBSOCKETS
# ---------------------------------------------------------------------------
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Stream new classifications to the client."""
    await ws.accept()
    log.info(f"WS /ws/live connected from {ws.client}")

    try:
        snapshot = list(state.recent)[-50:]
        await ws.send_json({"type": "snapshot", "results": snapshot})
    except Exception:
        return

    last_seen_id = state.total_processed
    interval = 1.0 / max(WEBSOCKET_BROADCAST_HZ, 1)
    try:
        while True:
            await asyncio.sleep(interval)
            current_id = state.total_processed
            if current_id > last_seen_id:
                diff = current_id - last_seen_id
                items = list(state.recent)
                new_items = items[-diff:] if diff <= len(items) else items
                if new_items:
                    await ws.send_json({"type": "update", "results": new_items})
                last_seen_id = current_id
    except WebSocketDisconnect:
        log.info("WS /ws/live disconnected")
    except Exception as e:
        log.exception(f"WS /ws/live error: {e}")


@app.websocket("/ws/stats")
async def ws_stats(ws: WebSocket):
    """Stream pipeline stats for the dashboard's metric tiles."""
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(1.0)
            payload = {
                "is_running": state.is_running,
                "capture_enabled": state.capture_enabled,
                "capture": state.capture.get_stats() if state.capture else None,
                "tracker": state.tracker.get_stats() if state.tracker else None,
                "predictor": state.predictor.get_stats() if state.predictor else None,
            }
            await ws.send_json(payload)
    except WebSocketDisconnect:
        log.info("WS /ws/stats disconnected")
    except Exception as e:
        log.exception(f"WS /ws/stats error: {e}")


# ---------------------------------------------------------------------------
# CLI ENTRY (`python -m backend.api.main`)
# ---------------------------------------------------------------------------
def serve() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", API_PORT))
    log.info(f"Starting server on {API_HOST}:{port}")
    uvicorn.run(
        "backend.api.main:app",
        host=API_HOST,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    serve()
