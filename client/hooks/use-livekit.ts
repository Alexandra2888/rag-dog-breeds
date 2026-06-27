"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  Room,
  RoomEvent,
  LocalAudioTrack,
  Track,
  createLocalAudioTrack,
} from "livekit-client";
import { createVoiceSession, endVoiceSession } from "@/lib/api-client";

interface UseLiveKitReturn {
  isConnected: boolean;
  isRecording: boolean;
  userSpeaking: boolean;
  aiSpeaking: boolean;
  error: string | null;
  toggleRecording: () => Promise<void>;
  connect: () => Promise<void>;
  disconnect: () => Promise<void>;
}

export function useLiveKit(): UseLiveKitReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const roomRef = useRef<Room | null>(null);
  const localAudioTrackRef = useRef<LocalAudioTrack | null>(null);
  const audioLevelIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const aiAudioCleanupRef = useRef<(() => void) | null>(null);
  const roomNameRef = useRef<string | null>(null);
  const agentConnectedRef = useRef<boolean>(false);

  const monitorAudioLevel = useCallback(
    (track: MediaStreamTrack, type: "user" | "ai"): (() => void) | null => {
      try {
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(
          new MediaStream([track])
        );
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const threshold = 30; // Adjust based on sensitivity needs

        const checkAudioLevel = () => {
          analyser.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;

          if (type === "user") {
            setUserSpeaking(average > threshold);
          } else {
            setAiSpeaking(average > threshold);
          }
        };

        const interval = setInterval(checkAudioLevel, 100);

        if (type === "user") {
          audioLevelIntervalRef.current = interval as unknown as NodeJS.Timeout;
        }

        return () => {
          clearInterval(interval);
          audioContext.close();
        };
      } catch (err) {
        console.error("Error setting up audio monitoring:", err);
        return null;
      }
    },
    []
  );

  const connect = useCallback(async () => {
    try {
      setError(null);

      // Fetch session info from Go backend
      const session = await createVoiceSession();
      roomNameRef.current = session.room_name;

      const room = new Room();
      roomRef.current = room;

      // Set up event listeners
      room.on(RoomEvent.Connected, () => {
        setIsConnected(true);
        console.log("Connected to LiveKit room:", session.room_name);
      });

      room.on(RoomEvent.Disconnected, () => {
        setIsConnected(false);
        setIsRecording(false);
        setUserSpeaking(false);
        setAiSpeaking(false);
        agentConnectedRef.current = false;
        console.log("Disconnected from LiveKit room");
      });

      room.on(RoomEvent.ParticipantConnected, (participant) => {
        console.log("Participant connected:", participant.identity);
        // Check if this is the AI agent (usually has "agent" in identity or name)
        if (
          participant.identity.includes("agent") ||
          participant.name?.toLowerCase().includes("agent")
        ) {
          agentConnectedRef.current = true;
          console.log("AI agent connected to room");
        }
      });

      room.on(RoomEvent.ParticipantDisconnected, (participant) => {
        console.log("Participant disconnected:", participant.identity);
        if (agentConnectedRef.current) {
          agentConnectedRef.current = false;
          console.log("AI agent disconnected from room");
        }
      });

      room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
        // Monitor AI participant audio. TrackSubscribed only fires for remote
        // participants, so any audio track here belongs to the AI agent.
        if (track.kind === "audio") {
          const mediaStreamTrack = track.mediaStreamTrack;
          if (mediaStreamTrack) {
            // Clean up previous AI audio monitoring if exists
            if (aiAudioCleanupRef.current) {
              aiAudioCleanupRef.current();
            }
            aiAudioCleanupRef.current = monitorAudioLevel(
              mediaStreamTrack,
              "ai"
            );
            console.log(
              "Subscribed to AI audio track from:",
              participant.identity
            );
          }
        }
      });

      // Connect using token from Go backend
      await room.connect(session.url, session.token);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to connect to LiveKit";
      setError(errorMessage);
      console.error("LiveKit connection error:", err);
    }
  }, [monitorAudioLevel]);

  const disconnect = useCallback(async () => {
    try {
      if (audioLevelIntervalRef.current) {
        clearInterval(audioLevelIntervalRef.current);
        audioLevelIntervalRef.current = null;
      }

      if (aiAudioCleanupRef.current) {
        aiAudioCleanupRef.current();
        aiAudioCleanupRef.current = null;
      }

      if (localAudioTrackRef.current) {
        localAudioTrackRef.current.stop();
        localAudioTrackRef.current = null;
      }

      if (roomRef.current) {
        await roomRef.current.disconnect();
        roomRef.current = null;
      }

      // Clean up session on backend (optional, but good practice)
      if (roomNameRef.current) {
        try {
          await endVoiceSession(roomNameRef.current);
        } catch (err) {
          console.warn("Failed to end voice session on backend:", err);
        }
        roomNameRef.current = null;
      }

      setIsConnected(false);
      setIsRecording(false);
      setUserSpeaking(false);
      setAiSpeaking(false);
      agentConnectedRef.current = false;
    } catch (err) {
      console.error("LiveKit disconnect error:", err);
    }
  }, []);

  const toggleRecording = useCallback(async () => {
    if (!roomRef.current) {
      await connect();
      return;
    }

    try {
      if (isRecording) {
        // Stop recording
        if (localAudioTrackRef.current) {
          await roomRef.current.localParticipant.unpublishTrack(
            localAudioTrackRef.current
          );
          localAudioTrackRef.current.stop();
          localAudioTrackRef.current = null;
        }

        if (audioLevelIntervalRef.current) {
          clearInterval(audioLevelIntervalRef.current);
          audioLevelIntervalRef.current = null;
        }

        setUserSpeaking(false);
        setIsRecording(false);
      } else {
        // Start recording
        const localTrack = await createLocalAudioTrack({
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        });

        localAudioTrackRef.current = localTrack;

        // Publish the track
        await roomRef.current.localParticipant.publishTrack(localTrack, {
          source: Track.Source.Microphone,
        });

        // Monitor user audio levels
        monitorAudioLevel(localTrack.mediaStreamTrack, "user");

        setIsRecording(true);
      }
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Failed to toggle recording";
      setError(errorMessage);
      console.error("Toggle recording error:", err);
    }
  }, [isRecording, connect, monitorAudioLevel]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    isConnected,
    isRecording,
    userSpeaking,
    aiSpeaking,
    error,
    toggleRecording,
    connect,
    disconnect,
  };
}
