"use client";

import { useState } from "react";
import { TextChat } from "@/components/text-chat";
import { VoiceChat } from "@/components/voice-chat";
import { MessageSquare, Mic } from "lucide-react";
import { cn } from "@/lib/utils";

type Mode = "text" | "voice";

export function ChatApp() {
  const [mode, setMode] = useState<Mode>("text");

  return (
    <div className="flex min-h-screen flex-col items-center bg-linear-to-br from-background via-background to-muted/20 p-4">
      <div className="flex w-full max-w-2xl flex-col items-center gap-6 py-8">
        <header className="text-center">
          <h1 className="text-2xl font-semibold">Dog Breed Assistant</h1>
          <p className="text-sm text-muted-foreground">
            Grounded in the dog breed book via RAG
          </p>
        </header>

        {/* Mode tabs */}
        <div className="inline-flex rounded-full border bg-card p-1">
          {(
            [
              { id: "text", label: "Text", icon: MessageSquare },
              { id: "voice", label: "Voice", icon: Mic },
            ] as const
          ).map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setMode(id)}
              className={cn(
                "flex items-center gap-2 rounded-full px-4 py-1.5 text-sm transition-colors",
                mode === id
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Active mode */}
      <div className="flex w-full flex-1 justify-center pb-8">
        {mode === "text" ? (
          <div className="flex h-[70vh] w-full max-w-2xl">
            <TextChat />
          </div>
        ) : (
          <VoiceChat />
        )}
      </div>
    </div>
  );
}
