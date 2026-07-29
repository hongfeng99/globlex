import { FormEvent, useState } from "react";
import { useAgentEvents } from "./useAgentEvents";

export function App() {
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const events = useAgentEvents(threadId);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const response = await fetch("/api/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const body = await response.json();
    setThreadId(body.thread_id);
  }

  return <main>
    <h1>Globex 跨境购物 Agent</h1>
    <form onSubmit={submit}>
      <textarea value={query} onChange={(e) => setQuery(e.target.value)}
        placeholder="告诉我你想买什么…" required />
      <button>开始</button>
    </form>
    <section>
      {events.map((event, index) =>
        <article key={index}><b>{event.message}</b>
          {event.event === "task_result" &&
            <pre>{String(event.data.final_answer ?? "")}</pre>}
        </article>)}
    </section>
  </main>;
}
