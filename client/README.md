# Dog Breed Assistant — Frontend

Next.js (App Router) frontend for the [Dog Breed RAG Assistant](../README.md). It
provides a **text chat** and a **voice chat** over the same RAG knowledge base.

- Text chat → calls the FastAPI backend `POST /query`, renders the grounded
  answer plus expandable source chunks.
- Voice chat → fetches a LiveKit room + token from `POST /api/voice/session`,
  connects with `@livekit/components-react`, and shows live assistant state
  (listening / thinking / speaking) with an audio-reactive visualizer.

## Stack

- Next.js 15 (App Router, React 19)
- Tailwind CSS v4, shadcn-style components, `lucide-react` icons
- `@livekit/components-react` + `livekit-client` for voice

## Getting started

```bash
bun install      # or: npm install / pnpm install
bun dev          # or: npm run dev   → http://localhost:3000
```

The backend must be running (see [`../server/README.md`](../server/README.md)).

## Configuration

Create `client/.env.local`:

```env
# Browser → FastAPI base URL
NEXT_PUBLIC_RAG_API_URL=http://localhost:8000
# Optional server-side (SSR) fallback
RAG_API_URL=http://localhost:8000
```

> Never put secrets behind `NEXT_PUBLIC_*` — those values ship to the browser.
> LiveKit tokens are minted server-side by the API (`/api/voice/session`); the
> client never holds the LiveKit secret.

## Structure

```
client/
├── app/
│   ├── layout.tsx          # root layout
│   ├── page.tsx            # renders <ChatApp/>
│   └── globals.css         # Tailwind v4 + theme tokens
├── components/
│   ├── chat-app.tsx        # text/voice mode tabs
│   ├── text-chat.tsx       # text Q&A UI (queryRag)
│   ├── voice-chat.tsx      # LiveKit room, mic controls, visualizer
│   └── ui/button.tsx       # shadcn-style button
└── lib/
    ├── api-client.ts       # queryRag / createVoiceSession / endVoiceSession
    ├── livekit-config.ts   # legacy/static config (deprecated; tokens come from API)
    └── utils.ts            # cn() helper
```

## Scripts

| Command | Action |
|---|---|
| `bun dev` | Dev server with hot reload |
| `bun run build` | Production build |
| `bun run start` | Serve the production build |
| `bun run lint` | ESLint |

## How it talks to the backend

`lib/api-client.ts` is the single integration point:

- `queryRag(query, topK)` → `POST /query` → `{ answer, chunks, cached }`
- `createVoiceSession(userId?)` → `POST /api/voice/session` → `{ room_name, token, url }`
- `endVoiceSession(roomName)` → `DELETE /api/voice/session/{room}`

See [`../docs/api-reference.md`](../docs/api-reference.md) for the full API.
