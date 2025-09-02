# enhanced_voice_rag_assistant.py

import os
import wave
import time
import threading
import re
import queue
from collections import deque
from typing import Iterator, Optional

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
from groq import Groq

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

# ============ THREAD SAFE STATE ============
class AppState:
    def __init__(self):
        self.is_listening = threading.Event()
        self.is_speaking = threading.Event()
        self.is_processing = threading.Event()
        self.stop_speaking = threading.Event()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()  # New queue for ready audio
        self.chat_history = []
        self.lock = threading.Lock()
    
    def reset_audio_flags(self):
        self.is_speaking.clear()
        self.stop_speaking.clear()
        # Clear both queues
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break

app_state = AppState()

# ============ EMBEDDINGS + DB SETUP ============
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ Vector database loaded successfully")
except Exception as e:
    print(f"⚠️  Vector database setup failed: {e}")
    retriever = None

# ============ LLM + WHISPER (Groq) ============
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY env var is required. Put it in .env")

try:
    chat_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.7)
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq LLM and Whisper initialized")
except Exception as e:
    print(f"❌ Groq initialization failed: {e}")
    raise

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly voice assistant. Have natural conversations while being helpful. "
    "When someone asks specific questions, use the provided context to give accurate answers. "
    "For casual conversation, respond naturally without forcing context usage. "
    "If you don't know something from the context, just say so simply. "
    "Keep responses conversational, warm, and concise - like talking to a friend. "
    "Avoid using special characters, markdown, or complex formatting in your responses since this is a voice interface."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """
Previous conversation:
{history}

Available information:
{context}

Current question: {question}

Respond naturally and conversationally.
""".strip()),
])

# ============ AUDIO UTILS ============
def list_input_devices():
    print("\n🎤 Available audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            default_mark = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default_mark}")
    print()

def calibrate_noise(seconds=2.0):
    """Calibrate noise threshold with better algorithm"""
    device = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX is not None else None
    print("🔧 Calibrating microphone... (stay quiet for a moment)")
    
    try:
        data = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16", device=device)
        sd.wait()
        
        # Calculate RMS with better noise estimation and filtering
        rms_values = []
        chunk_size = int(RATE * 0.1)  # 100ms chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            if len(chunk) > 0:
                rms = float(np.sqrt(np.mean(chunk.astype(np.int64) ** 2)))
                # Filter out very high spikes (likely noise)
                if rms < 5000:  # Reasonable upper limit
                    rms_values.append(rms)
        
        # Use percentile-based threshold for better noise handling
        if rms_values:
            # Use 75th percentile + margin for more stable threshold
            threshold_base = np.percentile(rms_values, 75)
            threshold = max(500.0, threshold_base * 3.0)  # More conservative
        else:
            threshold = 800.0
            
        print(f"🎯 Noise threshold set to: {threshold:.1f}")
        return threshold
        
    except Exception as e:
        print(f"⚠️  Calibration failed: {e}, using default threshold")
        return 800.0

def save_wav(filename, audio_bytes, samplerate=RATE):
    """Save audio bytes as WAV file"""
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_bytes)
        return True
    except Exception as e:
        print(f"❌ Failed to save audio: {e}")
        return False

def transcribe_audio_with_retry(file_path: str, max_retries: int = 2) -> str:
    """Transcribe audio with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            with open(file_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(file_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    language="en",
                    temperature=0.0,
                    response_format="text"
                )
            result = transcription.strip()
            
            # Filter out very short or nonsensical transcriptions
            if len(result) > 1 and result.lower().strip() not in ["", "you", "uh", "um"]:
                return result
            return ""
            
        except Exception as e:
            if attempt < max_retries:
                print(f"🔄 Transcription attempt {attempt + 1} failed, retrying...")
                time.sleep(0.5)
            else:
                print(f"❌ Transcription failed after {max_retries + 1} attempts: {e}")
                return ""

# ============ ENHANCED TTS WITH CONTINUOUS STREAMING ============
_tts_engine = None
_tts_worker_thread = None
_audio_player_thread = None
_tts_running = False

def _init_tts():
    """Initialize TTS engine with reliable fallback options"""
    global _tts_engine, TTS_BACKEND
    if _tts_engine is not None:
        return True
    
    # Try different TTS backends in order of preference
    backends_to_try = []
    
    if TTS_BACKEND == "auto":
        backends_to_try = ["edge", "gtts", "pyttsx3"]
    elif TTS_BACKEND == "edge":
        backends_to_try = ["edge", "gtts", "pyttsx3"]
    elif TTS_BACKEND == "gtts":
        backends_to_try = ["gtts", "edge", "pyttsx3"]
    elif TTS_BACKEND == "pyttsx3":
        backends_to_try = ["pyttsx3"]
    elif TTS_BACKEND == "coqui":
        backends_to_try = ["coqui", "edge", "gtts", "pyttsx3"]
    else:
        backends_to_try = ["pyttsx3"]
    
    for backend in backends_to_try:
        try:
            if backend == "edge":
                import edge_tts
                _tts_engine = edge_tts
                _tts_engine.backend_name = "edge"
                print("✅ Edge TTS initialized (Microsoft Neural Voices)")
                TTS_BACKEND = backend
                return True
                
            elif backend == "gtts":
                from gtts import gTTS
                import pygame
                pygame.mixer.init()
                _tts_engine = gTTS
                _tts_engine.backend_name = "gtts"
                print("✅ Google TTS initialized")
                TTS_BACKEND = backend
                return True
                
            elif backend == "pyttsx3":
                import pyttsx3
                _tts_engine = pyttsx3.init()
                _tts_engine.setProperty('rate', 220)  # Slightly faster
                _tts_engine.setProperty('volume', 0.9)
                _tts_engine.backend_name = "pyttsx3"
                print("✅ pyttsx3 TTS initialized")
                TTS_BACKEND = backend
                return True
                
            elif backend == "coqui":
                from TTS.api import TTS
                # Use a lightweight model to avoid conflicts
                _tts_engine = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                _tts_engine.backend_name = "coqui"
                print("✅ Coqui TTS initialized")
                TTS_BACKEND = backend
                return True
                
        except Exception as e:
            print(f"⚠️  {backend} TTS failed: {e}")
            continue
    
    print("❌ All TTS backends failed")
    return False

def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS synthesis"""
    # Remove markdown and special characters
    text = re.sub(r'[*_`#\[\]()]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Split text into speakable chunks at sentence boundaries"""
    # First split by sentences
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed chunk size, save current chunk
        if current_chunk and len(current_chunk + " " + sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk]

def _audio_player_worker():
    """Separate worker thread for playing audio continuously"""
    global _tts_running
    
    while _tts_running:
        try:
            # Get audio data from queue
            audio_data = app_state.audio_queue.get(timeout=0.1)
            
            if audio_data is None:  # Poison pill
                break
                
            if app_state.stop_speaking.is_set():
                continue
            
            # Play audio based on format
            try:
                app_state.is_speaking.set()
                
                if isinstance(audio_data, dict):
                    # Audio with metadata
                    if audio_data['format'] == 'wav_array':
                        sd.play(audio_data['data'], samplerate=audio_data['samplerate'])
                        sd.wait()
                    elif audio_data['format'] == 'wav_file':
                        import soundfile as sf
                        data, fs = sf.read(audio_data['file'])
                        sd.play(data, fs)
                        sd.wait()
                        # Clean up temp file
                        try:
                            os.unlink(audio_data['file'])
                        except:
                            pass
                elif isinstance(audio_data, str):
                    # Direct pyttsx3 text
                    if hasattr(_tts_engine, 'say'):
                        _tts_engine.say(audio_data)
                        _tts_engine.runAndWait()
                        
            except Exception as e:
                print(f"⚠️  Audio playback error: {e}")
                
        except queue.Empty:
            app_state.is_speaking.clear()
            continue
        except Exception as e:
            print(f"❌ Audio player error: {e}")
            break
    
    app_state.is_speaking.clear()
    print("🔇 Audio player stopped")

def _tts_worker():
    """Background worker for TTS synthesis - now focuses only on synthesis"""
    global _tts_running
    _tts_running = True
    
    while _tts_running:
        try:
            # Get text chunk from queue (with timeout)
            text_chunk = app_state.tts_queue.get(timeout=0.1)
            
            if text_chunk is None:  # Poison pill to stop worker
                break
                
            if app_state.stop_speaking.is_set():
                continue
            
            # Clean text
            clean_text = clean_text_for_tts(text_chunk)
            if not clean_text:
                continue
            
            # Synthesize audio (don't play here - just prepare)
            try:
                backend = getattr(_tts_engine, 'backend_name', 'unknown')
                
                if backend == "edge":
                    # Edge TTS implementation
                    import asyncio
                    import tempfile
                    
                    async def _edge_synthesize():
                        try:
                            # More robust Edge TTS with better error handling
                            communicate = _tts_engine.Communicate(
                                clean_text, 
                                "en-US-AriaNeural",
                                rate="+10%",  # Slightly faster speech
                                volume="+0%"
                            )
                            audio_data = b""
                            
                            async for chunk in communicate.stream():
                                if app_state.stop_speaking.is_set():
                                    break
                                if chunk["type"] == "audio" and chunk.get("data"):
                                    audio_data += chunk["data"]
                            
                            if audio_data and len(audio_data) > 100 and not app_state.stop_speaking.is_set():
                                # Save to temp file
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                    f.write(audio_data)
                                    temp_file = f.name
                                
                                # Queue for immediate playback
                                app_state.audio_queue.put({
                                    'format': 'wav_file',
                                    'file': temp_file
                                })
                            elif len(audio_data) <= 100:
                                # Skip very short audio (likely silence)
                                pass
                                
                        except Exception as e:
                            # Don't print error for very short texts or silence
                            if len(clean_text) > 3:
                                print(f"Edge TTS synthesis error: {e}")
                    
                    try:
                        asyncio.run(_edge_synthesize())
                    except Exception as e:
                        print(f"Edge TTS async error: {e}")
                        
                elif backend == "gtts":
                    # Google TTS implementation
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        tts = _tts_engine(text=clean_text, lang='en', slow=False)
                        tts.save(f.name)
                        temp_file = f.name
                    
                    if not app_state.stop_speaking.is_set():
                        app_state.audio_queue.put({
                            'format': 'wav_file',
                            'file': temp_file
                        })
                    
                elif backend == "coqui" and hasattr(_tts_engine, "tts"):
                    wav = _tts_engine.tts(clean_text)
                    if not app_state.stop_speaking.is_set():
                        app_state.audio_queue.put({
                            'format': 'wav_array',
                            'data': np.array(wav),
                            'samplerate': 22050
                        })
                        
                else:  # pyttsx3 or fallback
                    if not app_state.stop_speaking.is_set():
                        # For pyttsx3, we still need to play directly due to its nature
                        app_state.audio_queue.put(clean_text)
                        
            except Exception as e:
                print(f"⚠️  TTS synthesis error: {e}")
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ TTS synthesis worker error: {e}")
            break
    
    print("🔇 TTS synthesis worker stopped")

def start_tts_workers():
    """Start both TTS synthesis and audio playback workers"""
    global _tts_worker_thread, _audio_player_thread
    
    # Start synthesis worker
    if _tts_worker_thread is None or not _tts_worker_thread.is_alive():
        _tts_worker_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_worker_thread.start()
    
    # Start audio player worker
    if _audio_player_thread is None or not _audio_player_thread.is_alive():
        _audio_player_thread = threading.Thread(target=_audio_player_worker, daemon=True)
        _audio_player_thread.start()

def stop_tts_workers():
    """Stop both TTS workers"""
    global _tts_running
    _tts_running = False
    app_state.tts_queue.put(None)  # Poison pill for synthesis worker
    app_state.audio_queue.put(None)  # Poison pill for audio player

def speak_streaming(text: str):
    """Add text to TTS queue for immediate synthesis and playback"""
    if not text or not text.strip():
        return
    
    # Split text into smaller chunks for faster processing
    chunks = split_text_into_chunks(text, CHUNK_SIZE)
    
    # Add chunks to synthesis queue immediately
    for chunk in chunks:
        if chunk:
            app_state.tts_queue.put(chunk)

def interrupt_speech():
    """Interrupt current speech"""
    app_state.stop_speaking.set()
    app_state.reset_audio_flags()

# ============ ENHANCED AUDIO PROCESSING ============
class AudioProcessor:
    def __init__(self):
        self.energy_threshold = 600.0
        self.adaptive_threshold = True
        self.recent_energy = deque(maxlen=20)
        
    def calibrate(self, seconds=2.0):
        """Calibrate with adaptive threshold"""
        self.energy_threshold = calibrate_noise(seconds)
        
    def update_threshold(self, energy_level):
        """Dynamically update threshold based on recent audio"""
        if self.adaptive_threshold:
            self.recent_energy.append(energy_level)
            if len(self.recent_energy) >= 10:
                avg_recent = np.mean(list(self.recent_energy))
                self.energy_threshold = max(300.0, avg_recent * 2.5)

def enhanced_listen_and_transcribe():
    """Enhanced listening with better speech detection"""
    processor = AudioProcessor()
    processor.calibrate(1.5)
    
    device = int(INPUT_DEVICE_INDEX) if INPUT_DEVICE_INDEX is not None else None
    
    if HAVE_VAD:
        vad = webrtcvad.Vad(2)  # More aggressive VAD
    
    audio_bytes = bytearray()
    speech_started = False
    silence_count = 0
    speech_frame_count = 0
    blocksize = SAMPLES_PER_FRAME
    leftover = bytearray()
    
    print("🎧 Listening for speech... (speak now)")
    
    def process_frame(frame_bytes: bytes):
        nonlocal speech_started, silence_count, speech_frame_count
        
        # Calculate RMS energy
        audio_data = np.frombuffer(frame_bytes, dtype=np.int16)
        rms = float(np.sqrt(np.mean(audio_data.astype(np.int64) ** 2)))
        
        # Update adaptive threshold
        processor.update_threshold(rms)
        
        # Check for speech
        is_speech_energy = rms > processor.energy_threshold
        is_speech_flag = is_speech_energy
        
        # Use VAD if available
        if HAVE_VAD and len(frame_bytes) == BYTES_PER_FRAME:
            try:
                vad_result = vad.is_speech(frame_bytes, RATE)
                is_speech_flag = is_speech_energy or vad_result
            except:
                pass
        
        if DEBUG_AUDIO:
            print(f"Energy: {rms:.1f}, Threshold: {processor.energy_threshold:.1f}, Speech: {is_speech_flag}")
        
        if is_speech_flag:
            if not speech_started:
                speech_started = True
                print("🗣️  Speech detected, recording...")
            
            speech_frame_count += 1
            silence_count = 0
            audio_bytes.extend(frame_bytes)
            
        elif speech_started:
            # Add silence frames to audio for natural speech
            audio_bytes.extend(frame_bytes)
            silence_count += 1
            
            # End recording after enough silence
            if silence_count >= SILENCE_FRAMES and speech_frame_count >= MIN_SPEECH_FRAMES:
                print(f"✅ Recording complete ({len(audio_bytes)} bytes)")
                return True
        
        return False
    
    try:
        app_state.is_listening.set()
        with sd.InputStream(channels=1, samplerate=RATE, dtype="int16",
                          blocksize=blocksize, device=device) as stream:
            start_time = time.time()
            
            while app_state.is_listening.is_set():
                # Don't listen while speaking or processing
                if app_state.is_speaking.is_set() or app_state.is_processing.is_set():
                    time.sleep(0.1)
                    continue
                
                try:
                    data, overflowed = stream.read(blocksize)
                    if overflowed:
                        print("⚠️  Audio buffer overflow")
                    
                    chunk = data[:, 0].tobytes()
                    leftover.extend(chunk)
                    
                    # Process complete frames
                    while len(leftover) >= BYTES_PER_FRAME:
                        frame = bytes(leftover[:BYTES_PER_FRAME])
                        del leftover[:BYTES_PER_FRAME]
                        
                        if process_frame(frame):
                            raise StopIteration
                    
                    # Timeout protection
                    if speech_started and (time.time() - start_time > MAX_UTTERANCE_SEC):
                        print("⏰ Maximum utterance time reached")
                        break
                        
                except sd.CallbackAbort:
                    continue
                    
    except StopIteration:
        pass
    except Exception as e:
        print(f"❌ Audio input error: {e}")
    finally:
        app_state.is_listening.clear()
    
    if not audio_bytes or len(audio_bytes) < RATE:  # Less than 1 second
        return ""
    
    # Save and transcribe
    filename = f"temp_{int(time.time())}.wav"
    if not save_wav(filename, bytes(audio_bytes), RATE):
        return ""
    
    try:
        result = transcribe_audio_with_retry(filename, max_retries=2)
        os.remove(filename)  # Cleanup
        return result
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        try:
            os.remove(filename)
        except:
            pass
        return ""

# ============ ENHANCED RAG ============
def format_history(hist, max_turns=2):
    """Format chat history for context"""
    if not hist:
        return "This is the start of our conversation."
    
    recent = hist[-max_turns:]
    formatted = []
    for user_msg, assistant_msg in recent:
        formatted.append(f"You: {user_msg}")
        formatted.append(f"Assistant: {assistant_msg}")
    
    return "\n".join(formatted)

def get_streaming_rag_response(question: str) -> Iterator[str]:
    """Get streaming response from RAG system"""
    try:
        # Retrieve relevant documents
        context = ""
        if retriever:
            docs = retriever.invoke(question)
            if docs:
                context = "\n\n".join([d.page_content[:800] for d in docs[:3]])
        
        # Format history
        with app_state.lock:
            history_str = format_history(app_state.chat_history)
        
        # Create messages
        messages = prompt.format_messages(
            context=context, 
            question=question, 
            history=history_str
        )
        
        # Stream response
        accumulated_text = ""
        chunk_buffer = ""
        
        for chunk in chat_llm.stream(messages):
            if app_state.stop_speaking.is_set():
                break
                
            content = chunk.content
            if content:
                accumulated_text += content
                chunk_buffer += content
                
                # When we have enough text for a chunk, yield it
                if len(chunk_buffer) >= CHUNK_SIZE or content.endswith(('.', '!', '?', ',')):
                    yield chunk_buffer
                    chunk_buffer = ""
        
        # Yield any remaining text
        if chunk_buffer and not app_state.stop_speaking.is_set():
            yield chunk_buffer
            
        return accumulated_text
        
    except Exception as e:
        print(f"❌ RAG response error: {e}")
        yield "I apologize, but I encountered an error processing your request."

# ============ MAIN CONVERSATION LOOP ============
def handle_conversation():
    """Main conversation handling with streaming TTS"""
    while True:
        try:
            # Reset state
            app_state.reset_audio_flags()
            
            # Listen for user input
            print("\n" + "="*50)
            user_query = enhanced_listen_and_transcribe()
            
            if not user_query:
                print("❌ No speech detected, please try again...")
                continue
            
            print(f"👤 You: {user_query}")
            
            # Check for exit commands
            if user_query.lower().strip() in {"exit", "quit", "goodbye", "stop"}:
                farewell = "Goodbye! Have a great day!"
                print(f"🤖 Assistant: {farewell}")
                speak_streaming(farewell)
                # Wait for speech to complete
                while app_state.is_speaking.is_set() or not app_state.audio_queue.empty():
                    time.sleep(0.1)
                break
            
            # Process query
            app_state.is_processing.set()
            print("🤖 Assistant: ", end="", flush=True)
            
            full_response = ""
            
            # Get streaming response and speak it immediately
            for chunk in get_streaming_rag_response(user_query):
                if app_state.stop_speaking.is_set():
                    break
                
                print(chunk, end="", flush=True)
                full_response += chunk
                
                # Send chunk to TTS immediately for synthesis
                speak_streaming(chunk)
            
            print()  # New line after response
            app_state.is_processing.clear()
            
            # Wait for all TTS to finish before next iteration
            while not app_state.tts_queue.empty() or not app_state.audio_queue.empty() or app_state.is_speaking.is_set():
                time.sleep(0.05)  # Shorter sleep for better responsiveness
            
            # Store in history
            if full_response:
                with app_state.lock:
                    app_state.chat_history.append((user_query, full_response))
                    # Keep only recent history
                    if len(app_state.chat_history) > 10:
                        app_state.chat_history = app_state.chat_history[-10:]
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            interrupt_speech()
            break
        except Exception as e:
            print(f"❌ Conversation error: {e}")
            continue

# ============ MAIN FUNCTION ============
def main():
    """Main application entry point"""
    print("🚀 Enhanced Voice RAG Assistant Starting...")
    print("="*60)
    
    # List audio devices
    try:
        list_input_devices()
    except Exception as e:
        print(f"⚠️  Could not list audio devices: {e}")
    
    # Initialize TTS
    if not _init_tts():
        print("❌ Failed to initialize TTS, exiting...")
        return
    
    # Start TTS workers
    start_tts_workers()
    
    try:
        print("\n🎤 Voice Assistant Ready!")
        print("💡 Tips:")
        print("   • Speak clearly and wait for the beep")
        print("   • Say 'exit' or 'quit' to stop")
        print("   • The assistant will start speaking as soon as it generates text")
        print("="*60)
        
        # Start conversation loop
        handle_conversation()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        # Cleanup
        interrupt_speech()
        stop_tts_workers()
        print("👋 Voice Assistant stopped")

if __name__ == "__main__":
    main()