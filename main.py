"""
Enhanced Voice RAG Assistant
Main entry point for the application
"""

from audio_utils import list_input_devices
from tts_engine import init_tts_engine, start_tts_workers, stop_tts_workers, interrupt_speech
from conversation_handler import handle_conversation

def main():
    """Main application entry point"""
    print("🚀 Enhanced Voice RAG Assistant Starting...")
    print("="*60)

    try:
        list_input_devices()
    except Exception as e:
        print(f"⚠️  Could not list audio devices: {e}")

    if not init_tts_engine():
        print("❌ Failed to initialize TTS, exiting...")
        return

    start_tts_workers()

    try:
        print("\n🎤 Voice Assistant Ready!")
        print("💡 Tips:")
        print("   • Speak clearly and wait for the beep")
        print("   • Say 'exit' or 'quit' to stop")
        print("   • The assistant will start speaking as soon as it generates text")
        print("="*60)

        handle_conversation()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        interrupt_speech()
        stop_tts_workers()
        print("👋 Voice Assistant stopped")

if __name__ == "__main__":
    main()