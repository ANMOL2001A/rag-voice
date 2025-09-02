import os
import re
import time
import threading
import queue
import tempfile
import numpy as np
import sounddevice as sd

from config import TTS_BACKEND, CHUNK_SIZE
from app_state import app_state

_tts_engine = None
_tts_worker_thread = None
_audio_player_thread = None
_tts_running = False

def _init_tts():
    """Initialize TTS engine with reliable fallback options"""
    global _tts_engine, TTS_BACKEND
    if _tts_engine is not None:
        return True
    
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
                _tts_engine.setProperty('rate', 220) 
                _tts_engine.setProperty('volume', 0.9)
                _tts_engine.backend_name = "pyttsx3"
                print("✅ pyttsx3 TTS initialized")
                TTS_BACKEND = backend
                return True
                
            elif backend == "coqui":
                from TTS.api import TTS
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
    text = re.sub(r'[*_`#\[\]()]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Split text into speakable chunks at sentence boundaries"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if current_chunk and len(current_chunk + " " + sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()
    
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
    
    print(" TTS synthesis worker stopped")

def start_tts_workers():
    """Start both TTS synthesis and audio playback workers"""
    global _tts_worker_thread, _audio_player_thread
    
    if _tts_worker_thread is None or not _tts_worker_thread.is_alive():
        _tts_worker_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_worker_thread.start()
    
    if _audio_player_thread is None or not _audio_player_thread.is_alive():
        _audio_player_thread = threading.Thread(target=_audio_player_worker, daemon=True)
        _audio_player_thread.start()

def stop_tts_workers():
    """Stop both TTS workers"""
    global _tts_running
    _tts_running = False
    app_state.tts_queue.put(None) 
    app_state.audio_queue.put(None)  

def speak_streaming(text: str):
    """Add text to TTS queue for immediate synthesis and playback"""
    if not text or not text.strip():
        return
    
    chunks = split_text_into_chunks(text, CHUNK_SIZE)
    
    for chunk in chunks:
        if chunk:
            app_state.tts_queue.put(chunk)

def interrupt_speech():
    """Interrupt current speech"""
    app_state.stop_speaking.set()
    app_state.reset_audio_flags()

# Initialize TTS engine
def init_tts_engine():
    """Public function to initialize TTS engine"""
    return _init_tts()