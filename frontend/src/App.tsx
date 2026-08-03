import { FormEvent, useState } from "react";
import { useAgentEvents } from "./useAgentEvents";

export function App() {
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { events, connectionState } = useAgentEvents(threadId);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch("/api/task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const body = await response.json() as {
        thread_id?: string;
        detail?: string | Array<{ msg?: string }>;
      };
      if (!response.ok || !body.thread_id) {
        const detail = Array.isArray(body.detail)
          ? body.detail.map((item) => item.msg).filter(Boolean).join("；")
          : body.detail;
        throw new Error(detail || "任务创建失败");
      }
      setThreadId(body.thread_id);
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "任务创建失败",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return <main>
    <h1>Globex 跨境购物 Agent</h1>
    <form onSubmit={submit}>
      <textarea value={query} onChange={(e) => setQuery(e.target.value)}
        placeholder="告诉我你想买什么…" required />
      <button disabled={submitting}>
        {submitting ? "提交中…" : "开始"}
      </button>
    </form>
    {error && <p className="error">{error}</p>}
    {threadId && <p className="status">
      任务 {threadId} · 连接状态：{connectionState}
    </p>}
    <section>
      {events.map((event, index) =>
        <article key={index}><b>{event.message}</b>
          {event.event === "task_result" &&
            <pre>{String(event.data.final_answer ?? "")}</pre>}
          {event.event === "error" &&
            <pre className="error">{event.message}</pre>}
        </article>)}
    </section>
  </main>;
}
