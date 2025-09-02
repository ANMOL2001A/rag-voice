# config.py

import os
from dotenv import load_dotenv

# ============ LOAD .env ============
load_dotenv()

# ============ CONFIG ============
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Audio settings
RATE = 16000
FRAME_MS = int(os.getenv("FRAME_MS", "30"))
SAMPLES_PER_FRAME = int(RATE * FRAME_MS / 1000)
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "800"))  # Reduced for better responsiveness
SILENCE_FRAMES = max(2, END_SILENCE_MS // FRAME_MS)
MAX_UTTERANCE_SEC = int(os.getenv("MAX_UTTERANCE_SEC", "10"))  # Reduced
MIN_SPEECH_FRAMES = 3  # Minimum frames to consider speech
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX")
DEBUG_AUDIO = os.getenv("DEBUG_AUDIO", "0") == "1"

# TTS settings
TTS_BACKEND = os.getenv("TTS_BACKEND", "coqui")
CHUNK_SIZE = 30  # Reduced chunk size for faster processing

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly voice assistant. Have natural conversations while being helpful. "
    "When someone asks specific questions, use the provided context to give accurate answers. "
    "For casual conversation, respond naturally without forcing context usage. "
    "If you don't know something from the context, just say so simply. "
    "Keep responses conversational, warm, and concise - like talking to a friend. "
    "Avoid using special characters, markdown, or complex formatting in your responses since this is a voice interface."
)