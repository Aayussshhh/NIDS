import React, { useMemo, useState } from "react";
import { ChevronRight, AlertTriangle, Shield, Zap } from "lucide-react";
import "./LiveTraffic.css";

/*
 * The main event stream. Each row is one classified flow. Newest at top.
 *
 * Design choices:
 *   - Monospace columns with tabular-nums so addresses line up.
 *   - Color-coded left border indicates verdict severity at a glance.
 *   - Click a row to open ThreatDetails. The selected row stays
 *     highlighted so the operator can correlate multiple panels.
 *   - We cap the visible list at 200 to keep DOM operations cheap;
 *     older items are still in /api/recent if needed.
 */
export default function LiveTraffic({ events, selectedIndex, onSelect }) {
  const [filter, setFilter] = useState("all");

  // Filter the event stream client-side. Server-side filtering would
  // require API changes; this is fine for the modest event volumes we
  // expect on a single-NIC capture.
  const filtered = useMemo(() => {
    if (filter === "all") return events;
    if (filter === "attacks") return events.filter((e) => e.is_attack);
    if (filter === "anomalies")
      return events.filter((e) => e.is_anomaly || e.zero_day_override);
    if (filter === "benign") return events.filter((e) => !e.is_attack);
    return events;
  }, [events, filter]);

  return (
    <section className="live-traffic">
      <div className="live-traffic__header">
        <div className="live-traffic__title-block">
          <h2 className="live-traffic__title">
            <span className="serif">Live</span>{" "}
            <span className="mono">// classification stream</span>
          </h2>
          <span className="live-traffic__subtitle label">
            {filtered.length} of {events.length} events shown
          </span>
        </div>
        <div className="live-traffic__filters">
          {[
            ["all", "ALL"],
            ["attacks", "ATTACKS"],
            ["anomalies", "ANOMALIES"],
            ["benign", "BENIGN"],
          ].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setFilter(key)}
              className={`live-traffic__filter-btn ${
                filter === key ? "active" : ""
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="live-traffic__table-wrap">
        {/* Sticky-header table. */}
        <table className="live-traffic__table">
          <thead>
            <tr>
              <th className="col-time">TIME</th>
              <th className="col-verdict">VERDICT</th>
              <th className="col-conf">CONF</th>
              <th className="col-src">SOURCE</th>
              <th className="col-arrow"></th>
              <th className="col-dst">DESTINATION</th>
              <th className="col-proto">PROTO</th>
              <th className="col-bytes">BYTES</th>
              <th className="col-anom">AE</th>
              <th className="col-chevron"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan="10" className="live-traffic__empty">
                  No events. {events.length === 0 ? "Pipeline idle - press START to begin." : "Adjust filter to see more."}
                </td>
              </tr>
            ) : (
              filtered.map((evt, i) => {
                const idx = events.indexOf(evt);
                const selected = idx === selectedIndex;
                return (
                  <Row
                    key={`${evt.timestamp}-${i}`}
                    event={evt}
                    selected={selected}
                    onClick={() => onSelect(idx)}
                  />
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Row({ event, selected, onClick }) {
  const ts = new Date(event.timestamp * 1000);
  const time = ts.toLocaleTimeString("en-GB", { hour12: false });
  const ms = ts.getMilliseconds().toString().padStart(3, "0");

  // Verdict severity drives the row's left border color.
  let severity = "benign";
  if (event.zero_day_override) severity = "anomaly";
  else if (event.is_attack) severity = "malicious";
  else if (event.is_anomaly) severity = "suspicious";

  const Icon =
    severity === "malicious"
      ? AlertTriangle
      : severity === "anomaly"
      ? Zap
      : severity === "suspicious"
      ? AlertTriangle
      : Shield;

  const totalBytes =
    (event.flow.in_bytes || 0) + (event.flow.out_bytes || 0);

  return (
    <tr
      onClick={onClick}
      className={`live-traffic__row live-traffic__row--${severity} ${
        selected ? "selected" : ""
      }`}
    >
      <td className="col-time">
        {time}
        <span className="live-traffic__ms">.{ms}</span>
      </td>
      <td className="col-verdict">
        <span className={`pill ${severity}`}>
          <Icon size={9} />
          {event.classification}
        </span>
      </td>
      <td className="col-conf">
        <ConfidenceBar value={event.confidence} severity={severity} />
      </td>
      <td className="col-src">
        <span className="addr">{event.flow.src_ip}</span>
        <span className="port">:{event.flow.src_port}</span>
      </td>
      <td className="col-arrow">→</td>
      <td className="col-dst">
        <span className="addr">{event.flow.dst_ip}</span>
        <span className="port">:{event.flow.dst_port}</span>
      </td>
      <td className="col-proto">{event.flow.protocol}</td>
      <td className="col-bytes">{formatBytes(totalBytes)}</td>
      <td className="col-anom">
        <AnomalyIndicator score={event.anomaly_score} threshold={event.anomaly_threshold} />
      </td>
      <td className="col-chevron">
        <ChevronRight size={12} />
      </td>
    </tr>
  );
}

function ConfidenceBar({ value, severity }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="conf-bar" title={`${pct}%`}>
      <div className={`conf-bar__fill conf-bar__fill--${severity}`} style={{ width: `${pct}%` }} />
      <span className="conf-bar__text">{pct}</span>
    </div>
  );
}

function AnomalyIndicator({ score, threshold }) {
  const ratio = threshold > 0 ? score / threshold : 0;
  const above = score > threshold;
  const display = ratio >= 10 ? "10+" : ratio.toFixed(1);
  return (
    <span className={`anom-indicator ${above ? "anom-indicator--above" : ""}`}>
      ×{display}
    </span>
  );
}

function formatBytes(n) {
  if (n < 1024) return `${n}B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)}K`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)}M`;
  return `${(n / 1024 ** 3).toFixed(1)}G`;
}
