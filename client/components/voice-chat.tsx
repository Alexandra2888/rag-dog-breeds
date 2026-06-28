"use client";

import { useCallback, useState } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useVoiceAssistant,
  useMultibandTrackVolume,
  useLocalParticipant,
  useRoomContext,
  type AgentState,
} from "@livekit/components-react";
import { Button } from "@/components/ui/button";
import { createVoiceSession, endVoiceSession } from "@/lib/api-client";
import {
  Mic,
  MicOff,
  PhoneOff,
  Loader2,
  AlertCircle,
  AudioLines,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Session {
  url: string;
  token: string;
  roomName: string;
}

export function VoiceChat() {
  const [session, setSession] = useState<Session | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = useCallback(async () => {
    setError(null);
    setConnecting(true);
    try {
      const s = await createVoiceSession();
      setSession({ url: s.url, token: s.token, roomName: s.room_name });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start voice session");
    } finally {
      setConnecting(false);
    }
  }, []);

  const stop = useCallback(() => {
    setSession((current) => {
      if (current) endVoiceSession(current.roomName).catch(() => {});
      return null;
    });
  }, []);

  if (!session) {
    return <StartScreen onStart={start} connecting={connecting} error={error} />;
  }

  return (
    <LiveKitRoom
      serverUrl={session.url}
      token={session.token}
      connect
      audio
      video={false}
      onDisconnected={stop}
      onError={(e) => setError(e.message)}
      className="flex w-full flex-1 items-center justify-center"
    >
      <RoomAudioRenderer />
      <AssistantView />
    </LiveKitRoom>
  );
}

/* ------------------------------------------------------------------ */

function StartScreen({
  onStart,
  connecting,
  error,
}: {
  onStart: () => void;
  connecting: boolean;
  error: string | null;
}) {
  return (
    <div className="flex w-full max-w-md flex-col items-center gap-6 text-center">
      <div className="flex h-24 w-24 items-center justify-center rounded-full bg-primary/10 ring-1 ring-primary/20">
        <AudioLines className="h-10 w-10 text-primary" />
      </div>
      <div className="space-y-1">
        <h2 className="text-lg font-semibold">Talk to the breed assistant</h2>
        <p className="text-sm text-muted-foreground">
          Tap start and just speak. When you stop, the assistant answers out
          loud — grounded in the dog breed book.
        </p>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-destructive/10 px-4 py-2 text-left text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="whitespace-pre-wrap">{error}</span>
        </div>
      )}

      <Button
        size="lg"
        className="rounded-full px-8"
        onClick={onStart}
        disabled={connecting}
      >
        {connecting ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Connecting…
          </>
        ) : (
          <>
            <Mic className="mr-2 h-4 w-4" /> Start conversation
          </>
        )}
      </Button>
    </div>
  );
}

/* ------------------------------------------------------------------ */

const STATE_COPY: Record<string, { label: string; hint: string }> = {
  connecting: { label: "Connecting", hint: "Setting up the room…" },
  initializing: { label: "Waking up", hint: "The assistant is joining…" },
  listening: { label: "Listening", hint: "Ask about any dog breed" },
  thinking: { label: "Thinking", hint: "Searching the book…" },
  speaking: { label: "Speaking", hint: "Answering from the book" },
  disconnected: { label: "Disconnected", hint: "" },
};

function AssistantView() {
  const { state, audioTrack } = useVoiceAssistant();
  const room = useRoomContext();
  const { localParticipant } = useLocalParticipant();
  const [micOn, setMicOn] = useState(true);

  const toggleMic = useCallback(async () => {
    const next = !micOn;
    await localParticipant.setMicrophoneEnabled(next);
    setMicOn(next);
  }, [micOn, localParticipant]);

  const copy = STATE_COPY[state] ?? STATE_COPY.connecting;

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-8">
      {/* State pill */}
      <div
        className={cn(
          "flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-medium transition-colors",
          state === "speaking" && "bg-primary/15 text-primary",
          state === "thinking" && "bg-amber-500/15 text-amber-600",
          state === "listening" && "bg-emerald-500/15 text-emerald-600",
          (state === "connecting" || state === "initializing") &&
            "bg-muted text-muted-foreground"
        )}
      >
        <StateDot state={state} />
        {copy.label}
      </div>

      {/* Visualizer */}
      <Visualizer state={state} audioTrack={audioTrack} />

      <p className="h-5 text-sm text-muted-foreground">{copy.hint}</p>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <button
          onClick={toggleMic}
          className={cn(
            "flex h-12 w-12 items-center justify-center rounded-full border transition-colors",
            micOn
              ? "bg-background hover:bg-muted"
              : "border-destructive/40 bg-destructive/10 text-destructive"
          )}
          aria-label={micOn ? "Mute microphone" : "Unmute microphone"}
        >
          {micOn ? <Mic className="h-5 w-5" /> : <MicOff className="h-5 w-5" />}
        </button>

        <button
          onClick={() => room.disconnect()}
          className="flex h-12 w-12 items-center justify-center rounded-full bg-destructive text-destructive-foreground transition-colors hover:bg-destructive/90"
          aria-label="End conversation"
        >
          <PhoneOff className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}

function StateDot({ state }: { state: AgentState }) {
  const animate =
    state === "thinking" || state === "connecting" || state === "initializing";
  return (
    <span
      className={cn(
        "h-2 w-2 rounded-full",
        animate && "animate-pulse",
        state === "speaking" && "bg-primary",
        state === "thinking" && "bg-amber-500",
        state === "listening" && "bg-emerald-500",
        (state === "connecting" ||
          state === "initializing" ||
          state === "disconnected") &&
          "bg-muted-foreground"
      )}
    />
  );
}

/**
 * Audio-reactive bar visualizer. When the agent speaks, bar heights follow the
 * agent's audio. Otherwise bars idle-pulse to signal listening / thinking.
 */
function Visualizer({
  state,
  audioTrack,
}: {
  state: AgentState;
  audioTrack: ReturnType<typeof useVoiceAssistant>["audioTrack"];
}) {
  const BARS = 7;
  const volumes = useMultibandTrackVolume(audioTrack, { bands: BARS });
  const speaking = state === "speaking";

  return (
    <div className="flex h-36 items-end justify-center gap-2">
      {Array.from({ length: BARS }).map((_, i) => {
        const v = volumes[i] ?? 0;
        const height = speaking
          ? Math.max(10, Math.min(140, v * 320))
          : state === "thinking"
            ? 40
            : 14;
        return (
          <span
            key={i}
            className={cn(
              "w-3.5 rounded-full transition-[height] duration-100 ease-out",
              speaking
                ? "bg-primary"
                : state === "thinking"
                  ? "animate-pulse bg-amber-400"
                  : "animate-pulse bg-emerald-400/70"
            )}
            style={{
              height: `${height}px`,
              animationDelay: `${i * 90}ms`,
            }}
          />
        );
      })}
    </div>
  );
}
