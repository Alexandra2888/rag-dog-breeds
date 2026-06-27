"use client";

import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { queryRag, type ChunkResult } from "@/lib/api-client";
import { Send, Loader2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: ChunkResult[];
}

export function TextChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  const send = async () => {
    const query = input.trim();
    if (!query || loading) return;

    setError(null);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await queryRag(query);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: res.answer || "No answer returned.",
          sources: res.chunks,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
      requestAnimationFrame(() =>
        endRef.current?.scrollIntoView({ behavior: "smooth" })
      );
    }
  };

  return (
    <div className="flex h-full w-full max-w-2xl flex-col gap-4">
      {/* Messages */}
      <div className="flex-1 space-y-4 overflow-y-auto rounded-lg border bg-card/40 p-4">
        {messages.length === 0 && !loading && (
          <p className="py-12 text-center text-sm text-muted-foreground">
            Ask anything about dog breeds — e.g. &ldquo;What&rsquo;s a good
            apartment dog?&rdquo;
          </p>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              "flex",
              msg.role === "user" ? "justify-end" : "justify-start"
            )}
          >
            <div
              className={cn(
                "max-w-[85%] rounded-2xl px-4 py-2 text-sm",
                msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-foreground"
              )}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>

              {msg.sources && msg.sources.length > 0 && (
                <details className="mt-2 text-xs opacity-80">
                  <summary className="cursor-pointer select-none">
                    {msg.sources.length} source
                    {msg.sources.length > 1 ? "s" : ""}
                  </summary>
                  <ul className="mt-1 space-y-1">
                    {msg.sources.map((s) => (
                      <li key={s.id} className="border-l-2 pl-2">
                        {s.content.slice(0, 160)}
                        {s.content.length > 160 ? "…" : ""}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Thinking…</span>
          </div>
        )}

        <div ref={endRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-destructive/10 px-4 py-2 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="whitespace-pre-wrap">{error}</span>
        </div>
      )}

      {/* Input */}
      <form
        className="flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a dog breed…"
          className="flex-1 rounded-full border bg-background px-4 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
          disabled={loading}
        />
        <Button
          type="submit"
          size="icon"
          className="h-10 w-10 shrink-0 rounded-full"
          disabled={loading || !input.trim()}
        >
          <Send className="h-4 w-4" />
        </Button>
      </form>
    </div>
  );
}
