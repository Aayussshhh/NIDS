/*
 * Service layer for the FastAPI backend.
 *
 * Centralizes all HTTP and WebSocket calls so components don't reach
 * for fetch() directly. If we ever need to add auth headers or change
 * the base URL, there's one place to do it.
 *
 * In production, VITE_API_URL points to the Render backend URL.
 * In development, it's empty and the Vite proxy handles routing.
 */

// VITE_API_URL should be the full backend URL in production,
// e.g. "https://nids-hybrid-backend.onrender.com"
const API_BASE = import.meta.env.VITE_API_URL || "";

// WebSocket base: derive from API_BASE in production, or use current host in dev.
function getWsBase() {
  if (API_BASE) {
    // Convert https://foo.com -> wss://foo.com, http://foo.com -> ws://foo.com
    return API_BASE.replace(/^https:/, "wss:").replace(/^http:/, "ws:");
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
}
const WS_BASE = getWsBase();

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const api = {
  // ---------- REST ----------
  health: () => request("/"),
  listInterfaces: () => request("/api/interfaces"),
  getStats: () => request("/api/stats"),
  getRecent: (limit = 100) => request(`/api/recent?limit=${limit}`),
  getAttackTypes: () => request("/api/attack-types"),
  start: (interfaceName, bpfFilter = "ip", captureEnabled = true) =>
    request("/api/start", {
      method: "POST",
      body: JSON.stringify({
        interface: interfaceName,
        bpf_filter: bpfFilter,
        capture_enabled: captureEnabled,
      }),
    }),
  stop: () => request("/api/stop", { method: "POST" }),
  captureToggle: (enabled) =>
    request("/api/capture-toggle", {
      method: "POST",
      body: JSON.stringify({ enabled }),
    }),
  inject: (attackType, intensity = "medium") =>
    request("/api/inject", {
      method: "POST",
      body: JSON.stringify({ attack_type: attackType, intensity }),
    }),

  // ---------- WebSockets ----------
  openLiveStream: () => new WebSocket(`${WS_BASE}/ws/live`),
  openStatsStream: () => new WebSocket(`${WS_BASE}/ws/stats`),
};
