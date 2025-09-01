# voice_rag_assistant.py

import os
import subprocess
import wave
import time
import threading  # NEW: For async TTS
from collections import deque

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# ============ LOAD .env ============
load_dotenv()

# Optional TTS backends
TTS_BACKEND = os.getenv("TTS_BACKEND", "coqui")  # "coqui" or "pyttsx3"

# ============ CONFIG ============
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

WHISPER_CPP_PATH = os.getenv("WHISPER_CPP_PATH", "./whisper.cpp/main")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "./whisper.cpp/models/ggml-base.en.bin")

RATE = 16000  # required by webrtcvad
FRAME_MS = int(os.getenv("FRAME_MS", "20"))  # must be 10, 20, or 30 for webrtcvad
SAMPLES_PER_FRAME = int(RATE * FRAME_MS / 1000)  # 320 at 20ms
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # int16
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "800"))  # stop after this much silence
SILENCE_FRAMES = max(2, END_SILENCE_MS // FRAME_MS)
MAX_UTTERANCE_SEC = int(os.getenv("MAX_UTTERANCE_SEC", "20"))
INPUT_DEVICE_INDEX = os.getenv("INPUT_DEVICE_INDEX")  # optional explicit device id
DEBUG_AUDIO = os.getenv("DEBUG_AUDIO", "0") == "1"

# ============ EMBEDDINGS + DB SETUP ============
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ============ LLM (Groq) ============
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY env var is required. Put it in .env")
chat_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

SYSTEM_PROMPT = (
    "You are a helpful, warm RAG assistant. "
    "For specific questions, answer using the provided context. "
    "For casual greetings or general conversation, respond naturally and warmly. "
    "If a specific answer isn't in the context, say you don't have that information. "
    "Keep replies concise and conversational."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """
<chat_history>
{history}
</chat_history>

<context>
{context}
</context>

User: {question}
""".strip()),
])

chat_history = []  # list of (user, assistant) strings

# ============ AUDIO / VAD HELPERS ============
def list_input_devices():
    print("\nAvailable audio devices (index : name):")
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            print(f"  {i}: {d['name']}")
    print()


def calibrate_noise(seconds=0.6):
    """Capture a short sample to estimate ambient RMS and set dynamic threshold."""
    device = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX is not None else None
    data = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16", device=device)
    sd.wait()
    rms = float(np.sqrt(np.mean(data.astype(np.int64) ** 2)))
    threshold = max(500.0, rms * 2.5)
    if DEBUG_AUDIO:
        print(f"[Calibrate] ambient_rms={rms:.1f}, energy_threshold={threshold:.1f}")
    return threshold


def save_wav(filename, audio_bytes, samplerate=RATE):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_bytes)


def transcribe_audio(file_path: str) -> str:
    """Call whisper.cpp to transcribe file_path to text."""
    if not (os.path.exists(WHISPER_CPP_PATH) and os.path.exists(WHISPER_MODEL)):
        raise RuntimeError("Whisper.cpp binary/model not found. Set WHISPER_CPP_PATH and WHISPER_MODEL.")
    result = subprocess.run(
        [WHISPER_CPP_PATH, "-m", WHISPER_MODEL, "-f", file_path, "-otxt"],
        capture_output=True, text=True
    )
    txt_file = file_path + ".txt"
    if os.path.exists(txt_file):
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    return result.stdout.strip()


# ============ TTS (Coqui or fallback) ============
_tts_engine = None

def _init_tts():
    global _tts_engine
    if _tts_engine is not None:
        return
    if TTS_BACKEND == "coqui":
        try:
            from TTS.api import TTS
            _tts_engine = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
            return
        except Exception as e:
            print(f"[TTS] Coqui init failed: {e}. Falling back to pyttsx3.")
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
    except Exception as e:
        print(f"[TTS] pyttsx3 init failed: {e}. Audio output disabled.")


def clean_text_for_tts(text: str) -> str:
    """Clean text to avoid TTS character issues."""
    replacements = {
        "'": "'",  # smart apostrophe to regular
        "'": "'",  # another smart apostrophe  
        """: '"',  # smart quotes
        """: '"',
        "—": "-",  # em dash to hyphen
        "–": "-",  # en dash to hyphen
        "…": "...",  # ellipsis
        "‑": "-",  # non-breaking hyphen
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def speak_async(text: str):
    """Non-blocking TTS - starts speaking while continuing execution."""
    if not text:
        return
    
    def _speak_worker():
        if _tts_engine is None:
            _init_tts()
        
        # Clean text before TTS
        clean_text = clean_text_for_tts(text)
        
        if hasattr(_tts_engine, "tts"):  # Coqui API
            try:
                wav = _tts_engine.tts(clean_text)
                sd.play(np.array(wav), samplerate=22050)
                sd.wait()
            except Exception as e:
                print(f"[TTS] Coqui failed: {e}")
        else:  # pyttsx3
            try:
                _tts_engine.say(clean_text)
                _tts_engine.runAndWait()
            except Exception as e:
                print(f"[TTS] speak failed: {e}")
    
    # Start TTS in background thread
    tts_thread = threading.Thread(target=_speak_worker, daemon=True)
    tts_thread.start()


def speak(text: str):
    """Synchronous TTS - blocks until speech is complete."""
    if not text:
        return
    if _tts_engine is None:
        _init_tts()
    
    # Clean text before TTS
    clean_text = clean_text_for_tts(text)
    
    if hasattr(_tts_engine, "tts"):  # Coqui API
        try:
            wav = _tts_engine.tts(clean_text)
            sd.play(np.array(wav), samplerate=22050)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Coqui failed: {e}")
    else:  # pyttsx3
        try:
            _tts_engine.say(clean_text)
            _tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS] speak failed: {e}")


# ============ LISTEN LOOP WITH ROBUST VAD ============
def listen_and_transcribe():
    """Continuously read mic frames, detect end-of-utterance, return transcript."""
    energy_threshold = calibrate_noise(0.6)
    device = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX is not None else None
    if HAVE_VAD:
        vad = webrtcvad.Vad(2)
    ring_buffer = deque(maxlen=SILENCE_FRAMES)

    audio_bytes = bytearray()
    speech_started = False
    silence_count = 0
    frames_seen = 0
    blocksize = SAMPLES_PER_FRAME

    print("🎤 Listening... (speak now)")

    def process_frame(frame_bytes: bytes):
        nonlocal speech_started, silence_count, audio_bytes
        frame_np = np.frombuffer(frame_bytes, dtype=np.int16)
        rms = float(np.sqrt(np.mean(frame_np.astype(np.int64) ** 2)))
        is_speech_energy = rms > energy_threshold
        is_speech_flag = is_speech_energy
        if HAVE_VAD:
            try:
                is_speech_flag = is_speech_flag or vad.is_speech(frame_bytes, RATE)
            except Exception as e:
                if DEBUG_AUDIO:
                    print(f"[VAD] error: {e}")
        if DEBUG_AUDIO:
            print(f"[Frame] rms={rms:.0f} speech={is_speech_flag}")
        if is_speech_flag:
            speech_started = True
            silence_count = 0
            audio_bytes.extend(frame_bytes)
        else:
            if speech_started:
                silence_count += 1
                if silence_count >= SILENCE_FRAMES:
                    return True
        return False

    leftover = bytearray()

    try:
        with sd.InputStream(
            channels=1,
            samplerate=RATE,
            dtype="int16",
            blocksize=blocksize,
            device=device,
        ) as stream:
            start_time = time.time()
            while True:
                data, _ = stream.read(blocksize)
                chunk = data[:, 0].tobytes()
                leftover.extend(chunk)
                while len(leftover) >= BYTES_PER_FRAME:
                    frame = bytes(leftover[:BYTES_PER_FRAME])
                    del leftover[:BYTES_PER_FRAME]
                    frames_seen += 1
                    if process_frame(frame):
                        raise StopIteration
                if speech_started and (time.time() - start_time > MAX_UTTERANCE_SEC):
                    if DEBUG_AUDIO:
                        print("[Listen] Max utterance length reached; stopping.")
                    break
    except StopIteration:
        pass
    except Exception as e:
        print(f"[Audio] Input error: {e}")

    if not audio_bytes:
        return ""

    filename = "temp.wav"
    save_wav(filename, bytes(audio_bytes), RATE)
    return transcribe_audio(filename)


# ============ RAG + CHAT ============
def format_history(hist):
    if not hist:
        return "(no prior context)"
    return "\n".join([f"User: {u}\nAssistant: {a}" for u, a in hist[-6:]])


def rag_answer(question: str) -> str:
    # Handle casual greetings/conversations
    casual_responses = {
        "how are we doing": "We're doing great! I'm here and ready to help you with any questions.",
        "hello": "Hello! How can I help you today?",
        "hi": "Hi there! What would you like to know?",
        "how are you": "I'm doing well, thank you! How can I assist you?"
    }
    
    question_lower = question.lower().strip()
    for phrase, response in casual_responses.items():
        if phrase in question_lower:
            return response
            
    docs = retriever.invoke(question)
    if not docs:
        context = "(no documents retrieved)"
    else:
        parts = [d.page_content[:2000] for d in docs[:4]]
        context = "\n\n".join(parts)

    history_str = format_history(chat_history)
    messages = prompt.format_messages(context=context, question=question, history=history_str)
    resp = chat_llm.invoke(messages)
    return resp.content.strip()


# ============ MAIN LOOP ============
if __name__ == "__main__":
    try:
        list_input_devices()
    except Exception as e:
        print(f"[Audio] Could not list devices: {e}")

    print("🤖 Voice Assistant Ready (say 'exit' to quit)")

    while True:
        user_query = listen_and_transcribe()
        if not user_query:
            continue
        print(f"You: {user_query}")
        if user_query.lower().strip() in {"exit", "quit"}:
            break
        try:
            answer = rag_answer(user_query)
        except Exception as e:
            answer = f"I hit an error generating the answer: {e}"
        print(f"Assistant: {answer}")
        speak_async(answer)
        chat_history.append((user_query, answer))