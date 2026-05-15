import React, { useEffect, useState } from "react";
import { Activity, Power, Square, AlertTriangle, Shield, Radio, CircleOff } from "lucide-react";
import { api } from "../services/api";
import "./Header.css";

/*
 * Top status bar - always visible. Shows system identity, current pipeline
 * state, the live counters, the capture toggle, and the Start/Stop control.
 */
export default function Header({ stats, isRunning, captureEnabled, onStartStop, onCaptureToggle }) {
  const [interfaces, setInterfaces] = useState([]);
  const [selectedIface, setSelectedIface] = useState("");
  const [bpfFilter, setBpfFilter] = useState("ip");
  const [busy, setBusy] = useState(false);

  // Fetch available interfaces once on mount.
  useEffect(() => {
    api
      .listInterfaces()
      .then((r) => {
        setInterfaces(r.interfaces || []);
        if (r.interfaces?.length && !selectedIface) {
          const candidate =
            r.interfaces.find(
              (i) => i.address && i.address !== "0.0.0.0" && i.address !== "127.0.0.1"
            ) || r.interfaces[0];
          setSelectedIface(candidate.name);
        }
      })
      .catch(() => setInterfaces([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleStartStop = async () => {
    if (busy) return;
    setBusy(true);
    try {
      if (isRunning) {
        await api.stop();
      } else {
        await api.start(selectedIface || null, bpfFilter || "ip", captureEnabled);
      }
      onStartStop();
    } catch (e) {
      alert(`Pipeline error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleCaptureToggle = async () => {
    if (!isRunning || busy) return;
    setBusy(true);
    try {
      const res = await api.captureToggle(!captureEnabled);
      onCaptureToggle(res.capture_enabled);
    } catch (e) {
      alert(`Capture toggle error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const captured = stats?.capture?.captured ?? 0;
  const flowsActive = stats?.tracker?.active_flows ?? 0;
  const classifications = stats?.predictor?.classifications_made ?? 0;
  const attacks = stats?.predictor?.attacks_flagged ?? 0;
  const anomalies = stats?.predictor?.anomalies_flagged ?? 0;
  const pps = Math.round(stats?.capture?.pps ?? 0);

  return (
    <header className="header">
      {/* Identity block */}
      <div className="header__identity">
        <div className="header__brand">
          <span className="header__brand-mark">◢◣</span>
          <span className="header__brand-text">
            <span className="header__brand-name">NIDS</span>
            <span className="header__brand-sub">OPERATIONS CONSOLE</span>
          </span>
        </div>
        <div className="header__meta">
          <span className={`live-dot ${isRunning ? "" : "idle"}`} />
          <span className="header__meta-text">
            {isRunning ? "MONITORING" : "STANDBY"}
          </span>
          <span className="header__meta-divider">·</span>
          <span className="header__meta-text mono">
            {pps.toLocaleString()} pps
          </span>
        </div>
      </div>

      {/* Live counter strip */}
      <div className="header__counters">
        <Counter label="PACKETS" value={captured} icon={Activity} color="info" />
        <Counter label="ACTIVE FLOWS" value={flowsActive} icon={Activity} color="info" />
        <Counter label="CLASSIFIED" value={classifications} icon={Shield} color="benign" />
        <Counter label="ATTACKS" value={attacks} icon={AlertTriangle} color="malicious" highlight={attacks > 0} />
        <Counter label="ANOMALIES" value={anomalies} icon={AlertTriangle} color="anomaly" highlight={anomalies > 0} />
      </div>

      {/* Pipeline control */}
      <div className="header__control">
        <select
          value={selectedIface}
          onChange={(e) => setSelectedIface(e.target.value)}
          disabled={isRunning}
          className="header__select"
        >
          {interfaces.length === 0 ? (
            <option value="">no interfaces detected</option>
          ) : (
            interfaces.map((i) => (
              <option key={i.name} value={i.name}>
                {(i.address || "?").padEnd(15)} · {i.name.slice(0, 50)}
              </option>
            ))
          )}
        </select>
        <input
          type="text"
          value={bpfFilter}
          onChange={(e) => setBpfFilter(e.target.value)}
          placeholder="BPF filter"
          className="header__filter"
          disabled={isRunning}
        />

        {/* Capture toggle */}
        <button
          onClick={handleCaptureToggle}
          disabled={!isRunning || busy}
          className={`header__capture-toggle ${captureEnabled ? "on" : "off"}`}
          title={captureEnabled ? "Disable real-time capture" : "Enable real-time capture"}
        >
          {captureEnabled ? <Radio size={12} /> : <CircleOff size={12} />}
          <span>{captureEnabled ? "CAPTURE ON" : "CAPTURE OFF"}</span>
        </button>

        <button
          onClick={handleStartStop}
          disabled={busy || (!isRunning && !selectedIface)}
          className={isRunning ? "danger" : "primary"}
        >
          {isRunning ? <Square size={12} /> : <Power size={12} />}
          <span style={{ marginLeft: 6 }}>
            {busy ? "..." : isRunning ? "STOP" : "START"}
          </span>
        </button>
      </div>
    </header>
  );
}

function Counter({ label, value, icon: Icon, color, highlight }) {
  return (
    <div className={`counter ${highlight ? "counter--alert" : ""}`}>
      <div className="counter__head">
        <Icon size={11} className={`counter__icon counter__icon--${color}`} />
        <span className="counter__label">{label}</span>
      </div>
      <div className={`counter__value counter__value--${color}`}>
        {Number(value).toLocaleString()}
      </div>
    </div>
  );
}
