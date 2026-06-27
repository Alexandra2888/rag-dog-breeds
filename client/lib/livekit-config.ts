/**
 * LiveKit configuration
 * Note: URL and token are now fetched dynamically from Go backend
 * API key/secret are only needed for server-side operations
 */
export const LIVEKIT_CONFIG = {
  // Legacy static config (deprecated - use dynamic tokens from Go backend)
  url: process.env.NEXT_PUBLIC_LIVEKIT_URL || "",
  token: process.env.NEXT_PUBLIC_LIVEKIT_TOKEN || "",
  // Server-side only (for Go backend)
  apiKey: process.env.NEXT_PUBLIC_LIVEKIT_API_KEY || "",
  apiSecret: process.env.NEXT_PUBLIC_LIVEKIT_API_SECRET || "",
};

export const ROOM_NAME = "voice-chatbot";
