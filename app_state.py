import threading
import queue

class AppState:
    def __init__(self):
        self.is_listening = threading.Event()
        self.is_speaking = threading.Event()
        self.is_processing = threading.Event()
        self.stop_speaking = threading.Event()
        self.tts_queue = queue.Queue()
        self.audio_queue = queue.Queue()  
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