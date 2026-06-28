/**
 * API client for the FastAPI RAG backend (text query + LiveKit voice sessions).
 */

export interface VoiceSessionRequest {
  user_id?: string;
}

export interface VoiceSessionResponse {
  room_name: string;
  token: string;
  url: string;
}

export interface ChunkResult {
  id: string;
  content: string;
  similarity_score?: number | null;
  metadata?: Record<string, unknown> | null;
}

export interface QueryResponse {
  query: string;
  chunks: ChunkResult[];
  answer?: string | null;
  /** True when this answer was served from the backend cache (no LLM call). */
  cached?: boolean;
}

const getBackendUrl = (): string => {
  if (typeof window === "undefined") {
    return process.env.RAG_API_URL || "http://localhost:8000";
  }
  return process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8000";
};

const connectionError = (backendUrl: string): Error =>
  new Error(
    `Cannot connect to the RAG API at ${backendUrl}. Please ensure:\n` +
      `1. FastAPI is running on port 8000 (uvicorn src.main:app ...)\n` +
      `2. Postgres + Ollama are up and the PDF is ingested\n` +
      `3. NEXT_PUBLIC_RAG_API_URL is set correctly in .env.local`
  );

/**
 * Ask the RAG system a question and get a grounded answer plus source chunks.
 */
export async function queryRag(
  query: string,
  topK = 8
): Promise<QueryResponse> {
  const backendUrl = getBackendUrl();

  try {
    const response = await fetch(`${backendUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK, include_metadata: true }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Query failed (${response.status}): ${errorText}`);
    }

    return response.json() as Promise<QueryResponse>;
  } catch (error) {
    if (error instanceof TypeError && error.message === "Failed to fetch") {
      throw connectionError(backendUrl);
    }
    throw error;
  }
}

/**
 * Create a voice session: request a LiveKit room + access token from FastAPI.
 */
export async function createVoiceSession(
  userId?: string
): Promise<VoiceSessionResponse> {
  const backendUrl = getBackendUrl();

  try {
    const response = await fetch(`${backendUrl}/api/voice/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId || `user-${Date.now()}`,
      } as VoiceSessionRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();

      if (response.status === 404) {
        throw new Error(
          `Voice session endpoint not found at ${backendUrl}/api/voice/session. Ensure FastAPI is up to date.`
        );
      }

      throw new Error(
        `Failed to create voice session (${response.status}): ${errorText}`
      );
    }

    return response.json() as Promise<VoiceSessionResponse>;
  } catch (error) {
    if (error instanceof TypeError && error.message === "Failed to fetch") {
      throw connectionError(backendUrl);
    }
    throw error;
  }
}

/**
 * Ends a voice session (optional cleanup endpoint).
 */
export async function endVoiceSession(roomName: string): Promise<void> {
  const backendUrl = getBackendUrl();
  const response = await fetch(`${backendUrl}/api/voice/session/${roomName}`, {
    method: "DELETE",
  });

  if (!response.ok && response.status !== 404) {
    const errorText = await response.text();
    throw new Error(
      `Failed to end voice session: ${response.status} ${errorText}`
    );
  }
}
