"""LiveKit voice agent for dog-breed RAG, built on livekit-agents 1.x.

Speech (STT + TTS) runs through OpenAI; the LLM and the RAG retrieval run
locally on Ollama + pgvector. Retrieved context is injected into each turn via
``Agent.on_user_turn_completed`` so the local LLM answers grounded in the book.
"""
import asyncio
import logging

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai

from src.rag_service import RAGService
from src.embeddings import EmbeddingGenerator
from src.database import Database
from src.config import settings

logger = logging.getLogger(__name__)

# RAG service is initialized lazily (DB / Ollama may not be ready at import time).
_rag_service = None


def get_rag_service() -> RAGService:
    """Lazily build and cache the RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(EmbeddingGenerator(), Database())
        logger.info("RAG service initialized for voice agent")
    return _rag_service


SYSTEM_PROMPT = """You are a friendly voice assistant that answers questions about dog breeds.
You have a knowledge base drawn from a comprehensive dog breed book; relevant
excerpts are provided to you before each answer. Use them to answer accurately.
Keep replies concise and conversational since they will be spoken aloud.
If the provided context does not contain the answer, say you don't know rather
than inventing information."""


class DogBreedAgent(Agent):
    """Voice agent that grounds every answer in RAG-retrieved book context."""

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Retrieve relevant chunks and inject them before the LLM responds."""
        query = (new_message.text_content or "").strip()
        if not query:
            return

        try:
            rag = get_rag_service()
            # Vector search is blocking (psycopg2 + Ollama), so run it off the loop.
            result = await asyncio.to_thread(rag.search, query, 5)
            chunks = result.get("results", [])
            if not chunks:
                return

            context = "\n\n".join(
                f"[Page {c.get('metadata', {}).get('page_number', 'N/A')}] {c['content']}"
                for c in chunks
            )
            turn_ctx.add_message(
                role="assistant",
                content=(
                    "Relevant excerpts from the dog breed book to answer the "
                    f"user's question:\n{context}"
                ),
            )
            logger.info(f"Injected {len(chunks)} RAG chunks for query: {query!r}")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit job entrypoint: wire up the voice pipeline and start the session."""
    logger.info("Starting dog-breed RAG voice agent")
    await ctx.connect()

    if not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY is not set. OpenAI STT and TTS will fail. "
            "Set it in your environment or .env file."
        )

    # Ollama exposes an OpenAI-compatible endpoint at /v1 for the LLM.
    ollama_v1_url = settings.ollama_base_url.rstrip("/") + "/v1"

    # OpenAI STT/TTS read OPENAI_API_KEY from the environment (loaded from .env
    # in main()). Pass it explicitly only when present so the plugin's own
    # env-var fallback still works if it isn't.
    openai_kwargs = {"api_key": settings.openai_api_key} if settings.openai_api_key else {}

    session = AgentSession(
        # Speech-to-text via OpenAI (Whisper / gpt-4o transcription).
        stt=openai.STT(**openai_kwargs),
        # LLM served locally by Ollama (OpenAI-compatible endpoint).
        llm=openai.LLM.with_ollama(
            model=settings.ollama_chat_model,
            base_url=ollama_v1_url,
        ),
        # Text-to-speech via OpenAI.
        tts=openai.TTS(**openai_kwargs),
        # VAD: AgentSession uses the bundled Silero VAD by default.
    )

    await session.start(agent=DogBreedAgent(), room=ctx.room)

    # Greet the user so it's clear the agent is live.
    await session.generate_reply(
        instructions="Briefly greet the user and offer to answer questions about dog breeds."
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Load .env so the LiveKit CLI sees LIVEKIT_URL / LIVEKIT_API_KEY /
    # LIVEKIT_API_SECRET (and OPENAI_API_KEY) when run locally.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
