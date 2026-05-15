import React, { useEffect, useState, useCallback } from "react";
import Header from "./components/Header";
import LiveTraffic from "./components/LiveTraffic";
import ThreatDetails from "./components/ThreatDetails";
import AnalyticsStrip from "./components/AnalyticsStrip";
import AttackInjector from "./components/AttackInjector";
import { api } from "./services/api";
import { useWebSocket } from "./hooks/useWebSocket";
import "./App.css";

/*
 * Top-level layout:
 *
 *   ┌─────────────────────────── HEADER ───────────────────────────┐
 *   ├────────────────────────── INJECTOR ──────────────────────────┤
 *   ├─────────────────────────────┬────────────────────────────────┤
 *   │                             │                                │
 *   │       LIVE TRAFFIC          │       THREAT DETAILS           │
 *   │     (event stream)          │     (selected event panel)     │
 *   │                             │                                │
 *   ├─────────────────────────────┴────────────────────────────────┤
 *   │                  ANALYTICS STRIP (donut + timeline)          │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * State strategy:
 *   - `events` is the master event buffer, capped at 500.
 *   - `stats` is the pipeline metrics object updated 1Hz by /ws/stats.
 *   - `captureEnabled` tracks the real-time capture toggle independently.
 *   - `selectedIndex` tracks the row clicked in LiveTraffic.
 */
const MAX_EVENTS = 500;

export default function App() {
  const [events, setEvents] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [stats, setStats] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [captureEnabled, setCaptureEnabled] = useState(true);

  // ---- Live event stream
  const { lastMessage: liveMsg, status: liveStatus } = useWebSocket(
    api.openLiveStream,
    true
  );

  // ---- Stats stream
  const { lastMessage: statsMsg } = useWebSocket(api.openStatsStream, true);

  // Sync stats into local state.
  useEffect(() => {
    if (statsMsg) {
      setStats(statsMsg);
      setIsRunning(statsMsg.is_running);
      if (statsMsg.capture_enabled !== undefined) {
        setCaptureEnabled(statsMsg.capture_enabled);
      }
    }
  }, [statsMsg]);

  // Merge incoming events (snapshot or update) into the buffer.
  useEffect(() => {
    if (!liveMsg?.results) return;
    setEvents((prev) => {
      const next =
        liveMsg.type === "snapshot"
          ? [...liveMsg.results].reverse()
          : [...liveMsg.results.slice().reverse(), ...prev];
      return next.slice(0, MAX_EVENTS);
    });
  }, [liveMsg]);

  // Initial fetch of recent results.
  useEffect(() => {
    api
      .getRecent(50)
      .then((r) => {
        setEvents(r.results || []);
      })
      .catch(() => {});
    api.getStats().then(setStats).catch(() => {});
  }, []);

  const handleStartStop = useCallback(() => {
    api.getStats().then(setStats).catch(() => {});
  }, []);

  const handleCaptureToggle = useCallback((enabled) => {
    setCaptureEnabled(enabled);
  }, []);

  const selectedEvent =
    selectedIndex != null && events[selectedIndex] ? events[selectedIndex] : null;

  return (
    <div className="app">
      <Header
        stats={stats}
        isRunning={isRunning}
        captureEnabled={captureEnabled}
        onStartStop={handleStartStop}
        onCaptureToggle={handleCaptureToggle}
      />

      <main className="app__main">
        <AttackInjector isRunning={isRunning} />

        <div className="app__top">
          <LiveTraffic
            events={events}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
          />
          <ThreatDetails event={selectedEvent} />
        </div>
        <div className="app__bottom">
          <AnalyticsStrip events={events} />
        </div>
      </main>

      <footer className="app__footer">
        <div className="app__footer-left">
          <span className="label">WS</span>
          <span
            className={`app__footer-status app__footer-status--${liveStatus}`}
          >
            {liveStatus}
          </span>
          <span className="label">·</span>
          <span className="label">EVENTS BUFFER {events.length}/{MAX_EVENTS}</span>
          <span className="label">·</span>
          <span className={`label ${captureEnabled ? "capture-on" : "capture-off"}`}>
            CAPTURE {captureEnabled ? "ON" : "OFF"}
          </span>
        </div>
        <div className="app__footer-right">
          <span className="label">
            HYBRID NIDS v1.0 · XGBOOST + CNN-BiLSTM-ATTN + CONV-AE + SHAP
          </span>
        </div>
      </footer>
    </div>
  );
}
