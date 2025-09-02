# conversation_handler.py

import time
from app_state import app_state
from audio_utils import enhanced_listen_and_transcribe
from rag_system import get_streaming_rag_response
from tts_engine import speak_streaming, interrupt_speech

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
