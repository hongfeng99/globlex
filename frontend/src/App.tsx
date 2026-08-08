import { FormEvent, useState } from "react";
import { useAgentEvents } from "./useAgentEvents";

export function App() {
  const [query, setQuery] = useState("");
  const [activeRequest, setActiveRequest] = useState("");
  const [userId, setUserId] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { events, connectionState } = useAgentEvents(threadId);
  const latestResult = [...events].reverse().find(
    (event) => event.event === "task_result",
  )?.data.final_answer;
  const resultText = typeof latestResult === "string" ? latestResult : "";
  const needsClarification = Boolean(
    activeRequest && [
      "需要了解以下",
      "请提供",
      "预算范围",
      "使用场景",
      "为了帮您精准",
    ].some((marker) => resultText.includes(marker)),
  );

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const effectiveQuery = needsClarification
        ? `${activeRequest}\n\n用户补充信息：\n${query.trim()}`
        : query.trim();
      const response = await fetch("/api/task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: effectiveQuery,
          user_id: userId.trim() || undefined,
        }),
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
      setActiveRequest(effectiveQuery);
      setThreadId(body.thread_id);
      setQuery("");
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
    <p className="demo-notice">
      当前为离线演示模式：商品、平台、价格、库存和销量均为
      模拟数据，不代表电商平台实时信息，也不提供购买链接。
    </p>
    <form onSubmit={submit}>
      <input value={userId} onChange={(e) => setUserId(e.target.value)}
        placeholder="用户 ID（可选，用于保存和应用偏好）" />
      <textarea value={query} onChange={(e) => setQuery(e.target.value)}
        placeholder={needsClarification
          ? "补充回答上面的问题；系统会自动合并上一轮需求…"
          : "告诉我你想买什么…"} required />
      <button disabled={submitting}>
        {submitting ? "提交中…" : needsClarification ? "继续" : "开始"}
      </button>
    </form>
    {needsClarification && <aside className="continuation-notice">
      下一次提交会作为上一轮购物需求的补充，不会丢失“机械键盘”等上下文。
      <button type="button" className="text-button" onClick={() => {
        setActiveRequest("");
        setThreadId(null);
      }}>改为新需求</button>
    </aside>}
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
