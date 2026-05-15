import { useEffect, useRef, useState } from "react";

/*
 * Generic reconnecting WebSocket hook.
 *
 * Why custom and not a library: we want predictable behavior on
 * dev-server hot-reload (close the old socket cleanly), and we want
 * the *latest* JSON payload exposed as state without an unbounded
 * accumulator.
 *
 * Args:
 *   factoryFn: () => WebSocket - called to (re)open. We pass a factory
 *     instead of a URL so callers can use the api.js helpers.
 *   enabled: when false, no socket is opened. Lets the dashboard pause
 *     streaming without unmounting the component.
 *
 * Returns:
 *   { lastMessage, status }
 *     lastMessage: the latest parsed JSON payload (or null)
 *     status: "idle" | "connecting" | "open" | "closed"
 */
export function useWebSocket(factoryFn, enabled = true) {
  const [lastMessage, setLastMessage] = useState(null);
  const [status, setStatus] = useState("idle");
  const wsRef = useRef(null);
  const retryRef = useRef(0);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) wsRef.current.close();
      setStatus("idle");
      return;
    }

    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      setStatus("connecting");
      let ws;
      try {
        ws = factoryFn();
      } catch (e) {
        scheduleRetry();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setStatus("open");
        retryRef.current = 0; // reset backoff on success
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          setLastMessage(data);
        } catch {
          // Ignore non-JSON frames.
        }
      };

      ws.onerror = () => {
        // Don't try to recover here - the close event fires next and
        // owns the reconnect logic.
      };

      ws.onclose = () => {
        if (cancelled) return;
        setStatus("closed");
        scheduleRetry();
      };
    };

    const scheduleRetry = () => {
      // Exponential backoff capped at 8s. Prevents reconnect storms
      // when the API is genuinely down.
      const delay = Math.min(8000, 500 * 2 ** retryRef.current);
      retryRef.current += 1;
      timerRef.current = setTimeout(connect, delay);
    };

    connect();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  return { lastMessage, status };
}
