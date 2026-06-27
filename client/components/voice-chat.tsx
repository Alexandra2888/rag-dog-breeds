"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { WaveVisualization } from "@/components/wave-visualization";
import { useLiveKit } from "@/hooks/use-livekit";
import { Mic, MicOff, Wifi, WifiOff, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export function VoiceChat() {
  const {
    isConnected,
    isRecording,
    userSpeaking,
    aiSpeaking,
    error,
    toggleRecording,
    connect,
    disconnect,
  } = useLiveKit();

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-linear-to-br from-background via-background to-muted/20 p-4">
      <div className="flex w-full max-w-4xl flex-col items-center gap-8">
        {/* Connection Status */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          {isConnected ? (
            <>
              <Wifi className="h-4 w-4 text-green-500" />
              <span>Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="h-4 w-4 text-red-500" />
              <span>Disconnected</span>
            </>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 rounded-lg bg-destructive/10 px-4 py-2 text-sm text-destructive">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}

        {/* AI Wave Visualization */}
        <div className="flex flex-col items-center gap-4">
          <div className="text-sm font-medium text-muted-foreground">AI</div>
          <WaveVisualization
            isActive={aiSpeaking}
            variant="ai"
            className="min-h-[80px]"
          />
        </div>

        {/* Main Control Button */}
        <div className="flex flex-col items-center gap-4">
          <Button
            onClick={toggleRecording}
            disabled={!isConnected}
            size="lg"
            className={cn(
              "h-20 w-20 rounded-full transition-all",
              isRecording
                ? "bg-destructive hover:bg-destructive/90"
                : "bg-primary hover:bg-primary/90",
              !isConnected && "opacity-50 cursor-not-allowed"
            )}
          >
            {isRecording ? (
              <MicOff className="h-8 w-8" />
            ) : (
              <Mic className="h-8 w-8" />
            )}
          </Button>
          <p className="text-sm text-muted-foreground">
            {isRecording ? "Stop Recording" : "Start Recording"}
          </p>
        </div>

        {/* User Wave Visualization */}
        <div className="flex flex-col items-center gap-4">
          <div className="text-sm font-medium text-muted-foreground">You</div>
          <WaveVisualization
            isActive={userSpeaking && isRecording}
            variant="user"
            className="min-h-[80px]"
          />
        </div>
      </div>
    </div>
  );
}
