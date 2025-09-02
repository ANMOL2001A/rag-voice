import time
from app_state import app_state
from audio_utils import enhanced_listen_and_transcribe
from rag_system import get_streaming_rag_response
from tts_engine import speak_streaming, interrupt_speech

def handle_conversation():
    """Main conversation handling with streaming TTS"""
    while True:
        try:
            app_state.reset_audio_flags()
            
            print("\n" + "="*50)
            user_query = enhanced_listen_and_transcribe()
            
            if not user_query:
                print("No speech detected, please try again...")
                continue
            
            print(f" You: {user_query}")
            
            if user_query.lower().strip() in {"exit", "quit", "goodbye", "stop"}:
                farewell = "Goodbye! Have a great day!"
                print(f"Assistant: {farewell}")
                speak_streaming(farewell)
                while app_state.is_speaking.is_set() or not app_state.audio_queue.empty():
                    time.sleep(0.1)
                break
            
            app_state.is_processing.set()
            print("Assistant: ", end="", flush=True)
            
            full_response = ""
            
            sentence_buffer = ""
            sentence_enders = {'.', '!', '?', ',', ';', ':'}
            
            for chunk in get_streaming_rag_response(user_query):
                if app_state.stop_speaking.is_set():
                    break
                
                print(chunk, end="", flush=True)
                full_response += chunk
                sentence_buffer += chunk
                
                last_char_pos = -1
                for i in range(len(sentence_buffer) - 1, -1, -1):
                    if sentence_buffer[i] in sentence_enders:
                        last_char_pos = i
                        break
                
                if last_char_pos != -1:
                    sentence_to_speak = sentence_buffer[:last_char_pos + 1]
                    sentence_buffer = sentence_buffer[last_char_pos + 1:]
                    
                    speak_streaming(sentence_to_speak)

            if sentence_buffer.strip() and not app_state.stop_speaking.is_set():
                speak_streaming(sentence_buffer)
            
            print() 
            app_state.is_processing.clear()
            
            while not app_state.tts_queue.empty() or not app_state.audio_queue.empty() or app_state.is_speaking.is_set():
                time.sleep(0.05)  
            
            if full_response:
                with app_state.lock:
                    app_state.chat_history.append((user_query, full_response))
                    if len(app_state.chat_history) > 10:
                        app_state.chat_history = app_state.chat_history[-10:]
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            interrupt_speech()
            break
        except Exception as e:
            print(f"❌ Conversation error: {e}")
            continue
