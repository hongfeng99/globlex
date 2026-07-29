import { useEffect, useState } from "react";

export type AgentEvent = {
  event: string;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
};

export function useAgentEvents(threadId: string | null) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  useEffect(() => {
    if (!threadId) return;
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws/${threadId}`);
    ws.onmessage = ({ data }) => {
      setEvents((old) => [...old, JSON.parse(data) as AgentEvent]);
    };
    ws.onopen = () => ws.send("ready");
    return () => ws.close();
  }, [threadId]);
  return events;
}
