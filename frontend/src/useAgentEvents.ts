import { useEffect, useState } from "react";

export type AgentEvent = {
  event: string;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
};

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed";

export function useAgentEvents(threadId: string | null) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("idle");

  useEffect(() => {
    setEvents([]);
    if (!threadId) {
      setConnectionState("idle");
      return;
    }

    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const seen = new Set<string>();

    function connect() {
      setConnectionState((current) =>
        current === "open" ? "reconnecting" : "connecting",
      );
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(
        `${protocol}://${location.host}/ws/${threadId}`,
      );
      socket.onmessage = ({ data }) => {
        try {
          const event = JSON.parse(data) as AgentEvent;
          const key = `${event.timestamp}:${event.event}:${JSON.stringify(event.data)}`;
          if (seen.has(key)) return;
          seen.add(key);
          setEvents((old) => [...old, event]);
        } catch {
          // 忽略不符合 AgentEvent 契约的服务端消息。
        }
      };
      socket.onopen = () => {
        setConnectionState("open");
        socket?.send("ready");
      };
      socket.onclose = () => {
        if (disposed) {
          setConnectionState("closed");
          return;
        }
        setConnectionState("reconnecting");
        reconnectTimer = setTimeout(connect, 1500);
      };
      socket.onerror = () => socket?.close();
    }

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [threadId]);

  return { events, connectionState };
}
