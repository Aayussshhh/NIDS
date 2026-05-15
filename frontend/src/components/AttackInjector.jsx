import React, { useEffect, useState } from "react";
import {
  Flame, Zap, Search, KeyRound, Syringe,
  ChevronDown, ChevronUp, Loader2, CheckCircle2, AlertCircle,
} from "lucide-react";
import { api } from "../services/api";
import "./AttackInjector.css";

/*
 * Attack injection panel for demo/presentation.
 *
 * Each card represents an injectable attack type. Clicking INJECT
 * triggers a POST /api/inject request that feeds synthetic attack
 * traffic into the prediction pipeline. Works regardless of whether
 * real-time capture is enabled.
 */

const ICON_MAP = {
  flame: Flame,
  zap: Zap,
  search: Search,
  key: KeyRound,
  syringe: Syringe,
};

const INTENSITY_LABELS = {
  low: { label: "Low", desc: "~50% packet volume" },
  medium: { label: "Medium", desc: "Normal volume" },
  high: { label: "High", desc: "~200% packet volume" },
};

const COLOR_MAP = {
  DoS: "var(--clr-dos)",
  DDoS: "var(--clr-ddos)",
  Reconnaissance: "var(--clr-recon)",
  BruteForce: "var(--clr-brute)",
  Injection: "var(--clr-inject)",
};

export default function AttackInjector({ isRunning }) {
  const [attacks, setAttacks] = useState({});
  const [expanded, setExpanded] = useState(true);
  const [injecting, setInjecting] = useState({}); // key -> "loading"|"success"|"error"
  const [intensities, setIntensities] = useState({});

  useEffect(() => {
    api
      .getAttackTypes()
      .then((r) => {
        setAttacks(r.attack_types || {});
        const defaults = {};
        Object.keys(r.attack_types || {}).forEach((k) => (defaults[k] = "medium"));
        setIntensities(defaults);
      })
      .catch(() => {});
  }, []);

  const handleInject = async (key) => {
    if (!isRunning) return;
    setInjecting((p) => ({ ...p, [key]: "loading" }));
    try {
      await api.inject(key, intensities[key] || "medium");
      setInjecting((p) => ({ ...p, [key]: "success" }));
      setTimeout(() => setInjecting((p) => ({ ...p, [key]: null })), 2500);
    } catch (e) {
      console.error("Inject error:", e);
      setInjecting((p) => ({ ...p, [key]: "error" }));
      setTimeout(() => setInjecting((p) => ({ ...p, [key]: null })), 3000);
    }
  };

  const entries = Object.entries(attacks);
  if (entries.length === 0) return null;

  return (
    <section className="injector">
      <button
        className="injector__header"
        onClick={() => setExpanded((e) => !e)}
      >
        <div className="injector__header-left">
          <Syringe size={14} />
          <span className="injector__title">ATTACK INJECTOR</span>
          <span className="injector__subtitle">Demonstration Mode</span>
        </div>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {expanded && (
        <div className="injector__grid">
          {entries.map(([key, info]) => {
            const IconComp = ICON_MAP[info.icon] || AlertCircle;
            const status = injecting[key];
            const accent = COLOR_MAP[info.expected_class] || "var(--clr-accent)";

            return (
              <div
                key={key}
                className={`inject-card ${status === "loading" ? "inject-card--pulse" : ""}`}
                style={{ "--card-accent": accent }}
              >
                <div className="inject-card__top">
                  <div className="inject-card__icon" style={{ color: accent }}>
                    <IconComp size={18} />
                  </div>
                  <div className="inject-card__info">
                    <span className="inject-card__name">{info.name}</span>
                    <span className="inject-card__class">
                      Detected as: <strong>{info.expected_class}</strong>
                    </span>
                  </div>
                </div>

                <p className="inject-card__desc">{info.description}</p>

                <div className="inject-card__controls">
                  <div className="inject-card__intensity">
                    {Object.entries(INTENSITY_LABELS).map(([val, meta]) => (
                      <button
                        key={val}
                        className={`intensity-btn ${intensities[key] === val ? "active" : ""}`}
                        onClick={() =>
                          setIntensities((p) => ({ ...p, [key]: val }))
                        }
                        title={meta.desc}
                      >
                        {meta.label}
                      </button>
                    ))}
                  </div>

                  <button
                    className="inject-btn"
                    onClick={() => handleInject(key)}
                    disabled={!isRunning || status === "loading"}
                    style={{ borderColor: accent }}
                  >
                    {status === "loading" ? (
                      <Loader2 size={13} className="spin" />
                    ) : status === "success" ? (
                      <CheckCircle2 size={13} />
                    ) : status === "error" ? (
                      <AlertCircle size={13} />
                    ) : (
                      <Syringe size={13} />
                    )}
                    <span>
                      {status === "loading"
                        ? "INJECTING..."
                        : status === "success"
                        ? "DONE"
                        : status === "error"
                        ? "FAILED"
                        : "INJECT"}
                    </span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {!isRunning && expanded && (
        <div className="injector__disabled-notice">
          <AlertCircle size={13} />
          <span>Start the pipeline to enable attack injection</span>
        </div>
      )}
    </section>
  );
}
