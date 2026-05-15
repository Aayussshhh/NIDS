import React from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Cell } from "recharts";
import { ShieldAlert, Brain, Network, Sparkles } from "lucide-react";
import "./ThreatDetails.css";

/*
 * Right-side detail panel that surfaces "WHY did the model say that?"
 *
 * Sections:
 *   1. Verdict header - what the meta-learner decided + confidence
 *   2. Model signals - what each base learner said (transparency into
 *      the ensemble decision)
 *   3. Class probability distribution - bar chart of all 9 classes
 *   4. Top features (SHAP) - which inputs drove the decision
 *   5. Flow metadata - 5-tuple, byte counts, timing
 *
 * If no flow is selected, show an empty-state guide.
 */
export default function ThreatDetails({ event }) {
  if (!event) {
    return <EmptyState />;
  }

  const probData = Object.entries(event.class_probabilities)
    .map(([name, value]) => ({ name, value: value * 100 }))
    .sort((a, b) => b.value - a.value);

  let severity = "benign";
  if (event.zero_day_override) severity = "anomaly";
  else if (event.is_attack) severity = "malicious";
  else if (event.is_anomaly) severity = "suspicious";

  return (
    <aside className="threat-details">
      {/* ----- VERDICT HEADER ----- */}
      <div className={`threat-details__verdict threat-details__verdict--${severity}`}>
        <div className="threat-details__verdict-label">VERDICT</div>
        <div className="threat-details__verdict-text">
          {event.classification.toUpperCase()}
        </div>
        <div className="threat-details__verdict-conf">
          confidence{" "}
          <span className="threat-details__verdict-conf-num">
            {(event.confidence * 100).toFixed(1)}%
          </span>
        </div>
        {event.zero_day_override && (
          <div className="threat-details__zero-day-flag">
            <Sparkles size={11} />
            ZERO-DAY OVERRIDE · autoencoder dissented from supervised models
          </div>
        )}
      </div>

      {/* ----- FLOW META ----- */}
      <Section title="Flow / 5-tuple" icon={Network}>
        <KV label="Source">
          <span className="mono">
            {event.flow.src_ip}:{event.flow.src_port}
          </span>
        </KV>
        <KV label="Destination">
          <span className="mono">
            {event.flow.dst_ip}:{event.flow.dst_port}
          </span>
        </KV>
        <KV label="Protocol">{event.flow.protocol}</KV>
        <KV label="Duration">{event.flow.duration_ms.toFixed(1)} ms</KV>
        <KV label="First seen">
          {new Date(event.flow.first_seen * 1000).toLocaleTimeString()}
        </KV>
      </Section>

      {/* ----- MODEL SIGNALS ----- */}
      <Section title="Ensemble breakdown" icon={Brain}>
        <SignalRow
          name="XGBoost"
          prediction={event.model_signals.xgboost.prediction}
          confidence={event.model_signals.xgboost.confidence}
        />
        <SignalRow
          name="CNN+BiLSTM+Attn"
          prediction={event.model_signals.deep_model.prediction}
          confidence={event.model_signals.deep_model.confidence}
        />
        <div className="signal-row">
          <div className="signal-row__name">Autoencoder</div>
          <div className="signal-row__value">
            <span className="mono">
              MSE {event.model_signals.autoencoder.reconstruction_error.toExponential(2)}
            </span>
            <span
              className={
                event.model_signals.autoencoder.is_anomaly
                  ? "signal-row__tag signal-row__tag--alert"
                  : "signal-row__tag"
              }
            >
              {event.model_signals.autoencoder.is_anomaly ? "ANOMALOUS" : "NORMAL"}
            </span>
          </div>
          <div className="signal-row__bar">
            <div
              className="signal-row__bar-fill"
              style={{
                width: `${Math.min(
                  100,
                  (event.model_signals.autoencoder.reconstruction_error /
                    event.model_signals.autoencoder.threshold) *
                    50
                )}%`,
                background: event.model_signals.autoencoder.is_anomaly
                  ? "var(--status-anomaly)"
                  : "var(--text-muted)",
              }}
            />
            <div
              className="signal-row__bar-threshold"
              style={{ left: "50%" }}
              title="Anomaly threshold"
            />
          </div>
        </div>
      </Section>

      {/* ----- CLASS PROBABILITY DISTRIBUTION ----- */}
      <Section title="Class distribution" icon={ShieldAlert}>
        <div className="threat-details__chart">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={probData}
              layout="vertical"
              margin={{ top: 0, right: 24, left: 0, bottom: 0 }}
            >
              <XAxis
                type="number"
                domain={[0, 100]}
                hide
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 10, fill: "#8a96a3", fontFamily: "JetBrains Mono" }}
                width={90}
                axisLine={false}
                tickLine={false}
              />
              <Bar dataKey="value" radius={[0, 0, 0, 0]}>
                {probData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={
                      entry.name === event.classification
                        ? severity === "malicious"
                          ? "#ef4444"
                          : severity === "anomaly"
                          ? "#a78bfa"
                          : "#4ade80"
                        : "#1f2831"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* ----- SHAP FEATURES ----- */}
      {event.top_features && event.top_features.length > 0 && (
        <Section title="Top contributing features (SHAP)" icon={Sparkles}>
          {event.top_features.map((f, i) => (
            <div className="feature-row" key={i}>
              <span className="feature-row__name mono">{f.name}</span>
              <span className="feature-row__value">
                {f.shap_value > 0 ? "+" : ""}
                {f.shap_value.toFixed(3)}
              </span>
              <div className="feature-row__bar">
                <div
                  className="feature-row__bar-fill"
                  style={{
                    width: `${Math.min(
                      100,
                      Math.abs(f.shap_value) * 200
                    )}%`,
                    background:
                      f.shap_value > 0
                        ? "var(--status-malicious)"
                        : "var(--status-benign)",
                  }}
                />
              </div>
            </div>
          ))}
        </Section>
      )}
    </aside>
  );
}

function Section({ title, icon: Icon, children }) {
  return (
    <div className="threat-details__section">
      <div className="threat-details__section-title">
        <Icon size={11} />
        <span>{title}</span>
      </div>
      <div className="threat-details__section-body">{children}</div>
    </div>
  );
}

function KV({ label, children }) {
  return (
    <div className="kv">
      <span className="kv__label">{label}</span>
      <span className="kv__value">{children}</span>
    </div>
  );
}

function SignalRow({ name, prediction, confidence }) {
  const pct = (confidence * 100).toFixed(1);
  return (
    <div className="signal-row">
      <div className="signal-row__name">{name}</div>
      <div className="signal-row__value">
        <span className="mono">{prediction}</span>
        <span className="signal-row__tag">{pct}%</span>
      </div>
      <div className="signal-row__bar">
        <div className="signal-row__bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <aside className="threat-details threat-details--empty">
      <div className="threat-details__empty-mark">⌖</div>
      <div className="threat-details__empty-title serif">
        <em>Select an event</em>
      </div>
      <div className="threat-details__empty-text">
        Click any row in the live stream to inspect the ensemble's reasoning,
        per-model votes, SHAP attributions, and the underlying flow metadata.
      </div>
    </aside>
  );
}
