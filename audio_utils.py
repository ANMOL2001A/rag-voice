# audio_utils.py

import os
import wave
import time
import numpy as np
import sounddevice as sd
from collections import deque
from groq import Groq

try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

from config import *
from app_state import app_state

# ============ GROQ CLIENT ============
groq_client = Groq(api_key=GROQ_API_KEY)

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
        return ""# audio_utils.py

import os
import wave
import time
import numpy as np
import sounddevice as sd
from collections import deque
from groq import Groq

try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

from config import *
from app_state import app_state

# ============ GROQ CLIENT ============
groq_client = Groq(api_key=GROQ_API_KEY)

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