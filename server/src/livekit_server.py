"""Standalone LiveKit agent server entrypoint.

LiveKit credentials are read from the environment by the worker:
LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.
"""
from src.livekit_agent import main

if __name__ == "__main__":
    main()
