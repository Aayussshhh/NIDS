import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import "./AnalyticsStrip.css";

/*
 * Bottom strip with two visualizations:
 *   1. Per-class event count (donut chart) - quick read on what
 *      attack types dominate the recent stream
 *   2. Events-per-minute timeline (area chart) - detect bursts and
 *      sustained attacks over time
 *
 * Both are computed client-side from the existing event buffer; no
 * extra API calls needed.
 */

const CLASS_COLORS = {
  Benign: "#4ade80",
  DoS: "#ef4444",
  DDoS: "#dc2626",
  Reconnaissance: "#fbbf24",
  BruteForce: "#f97316",
  Injection: "#ec4899",
  Botnet: "#a855f7",
  Exfiltration: "#a78bfa",
  Other: "#38bdf8",
};

export default function AnalyticsStrip({ events }) {
  // --- Per-class donut data
  const classCounts = useMemo(() => {
    const counts = {};
    for (const e of events) {
      counts[e.classification] = (counts[e.classification] || 0) + 1;
    }
    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [events]);

  // --- Events-per-minute timeline (last 30 minutes, 30 buckets)
  const timeline = useMemo(() => {
    const now = Date.now() / 1000;
    const buckets = 30;
    const bucketSec = 60;
    const data = Array.from({ length: buckets }, (_, i) => ({
      t: i,
      label: `-${buckets - i}m`,
      total: 0,
      attacks: 0,
      anomalies: 0,
    }));
    for (const e of events) {
      const age = now - e.timestamp;
      const bucket = buckets - 1 - Math.floor(age / bucketSec);
      if (bucket < 0 || bucket >= buckets) continue;
      data[bucket].total += 1;
      if (e.is_attack) data[bucket].attacks += 1;
      if (e.is_anomaly) data[bucket].anomalies += 1;
    }
    return data;
  }, [events]);

  return (
    <div className="analytics-strip">
      {/* DONUT */}
      <div className="analytics-strip__panel">
        <div className="analytics-strip__title">
          <span className="serif">Distribution</span>
          <span className="label">last {events.length} events</span>
        </div>
        <div className="analytics-strip__chart">
          {classCounts.length === 0 ? (
            <div className="analytics-strip__empty">no data</div>
          ) : (
            <ResponsiveContainer width="100%" height={140}>
              <PieChart>
                <Pie
                  data={classCounts}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={32}
                  outerRadius={56}
                  paddingAngle={2}
                  stroke="none"
                >
                  {classCounts.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={CLASS_COLORS[entry.name] || "#5b6672"}
                    />
                  ))}
                </Pie>
                <Tooltip
                  content={<CustomTooltip />}
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
          <div className="analytics-strip__legend">
            {classCounts.slice(0, 8).map((c) => (
              <div className="legend-item" key={c.name}>
                <span
                  className="legend-item__swatch"
                  style={{ background: CLASS_COLORS[c.name] || "#5b6672" }}
                />
                <span className="legend-item__name">{c.name}</span>
                <span className="legend-item__count">{c.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* TIMELINE */}
      <div className="analytics-strip__panel analytics-strip__panel--wide">
        <div className="analytics-strip__title">
          <span className="serif">Activity</span>
          <span className="label">events / minute · last 30 min</span>
        </div>
        <div className="analytics-strip__chart">
          <ResponsiveContainer width="100%" height={140}>
            <AreaChart data={timeline} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="totalGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="atkGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 9, fill: "#5b6672", fontFamily: "JetBrains Mono" }}
                axisLine={false}
                tickLine={false}
                interval={5}
              />
              <YAxis
                tick={{ fontSize: 9, fill: "#5b6672", fontFamily: "JetBrains Mono" }}
                axisLine={false}
                tickLine={false}
                width={28}
              />
              <Tooltip
                content={<CustomTooltip />}
                cursor={{ stroke: "rgba(255,255,255,0.1)", strokeWidth: 1 }}
              />
              <Area
                type="monotone"
                dataKey="total"
                stroke="#38bdf8"
                strokeWidth={1.5}
                fill="url(#totalGrad)"
              />
              <Area
                type="monotone"
                dataKey="attacks"
                stroke="#ef4444"
                strokeWidth={1.5}
                fill="url(#atkGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      {label && <div className="custom-tooltip__label">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="custom-tooltip__row">
          <span
            className="custom-tooltip__swatch"
            style={{ background: p.color || p.payload?.fill || "#fff" }}
          />
          <span className="custom-tooltip__name">{p.name}</span>
          <span className="custom-tooltip__value">{p.value}</span>
        </div>
      ))}
    </div>
  );
}
