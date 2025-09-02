# Enhanced Voice RAG Assistant

A voice-enabled RAG (Retrieval-Augmented Generation) assistant that allows natural conversation with your documents through speech.

## Project Structure

```
voice_rag_assistant/
├── main.py                    # Main entry point - run this file
├── config.py                  # Configuration and environment variables
├── app_state.py              # Thread-safe application state management
├── audio_utils.py            # Audio processing, recording, and transcription
├── tts_engine.py             # Text-to-speech engine and workers
├── rag_system.py             # RAG system with vector database and LLM
├── conversation_handler.py   # Main conversation loop logic
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (create this)
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file:**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   CHROMA_DB_DIR=./chroma_db
   TTS_BACKEND=edge
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Module Descriptions

### `main.py`
- Application entry point
- Initializes all components
- Handles startup/shutdown

### `config.py`
- All configuration variables
- Environment variable loading
- Audio and TTS settings

### `app_state.py`
- Thread-safe state management
- Event coordination between components
- Queue management for audio processing

### `audio_utils.py`
- Audio recording and processing
- Speech detection with VAD
- Whisper transcription via Groq
- Microphone calibration

### `tts_engine.py`
- Text-to-speech synthesis
- Multiple TTS backend support (Edge, gTTS, pyttsx3, Coqui)
- Streaming audio playback
- Worker thread management

### `rag_system.py`
- Vector database integration
- Document retrieval
- LLM integration with Groq
- Streaming response generation

### `conversation_handler.py`
- Main conversation loop
- User input processing
- Response coordination
- Chat history management

## Features

- **Voice Input**: Real-time speech recognition using Whisper
- **Voice Output**: Multiple TTS backends with streaming playback
- **RAG Integration**: Query your document collection through voice
- **Adaptive Audio**: Dynamic noise threshold adjustment
- **Streaming Responses**: Start speaking as soon as text is generated
- **Thread Safety**: Concurrent audio processing and synthesis

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Required: Your Groq API key |
| `CHROMA_DB_DIR` | `./chroma_db` | Vector database directory |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq LLM model |
| `TTS_BACKEND` | `coqui` | TTS backend (edge/gtts/pyttsx3/coqui/auto) |
| `FRAME_MS` | `30` | Audio frame duration |
| `END_SILENCE_MS` | `800` | Silence duration to end recording |
| `MAX_UTTERANCE_SEC` | `10` | Maximum recording duration |
| `INPUT_DEVICE_INDEX` | auto | Microphone device index |
| `DEBUG_AUDIO` | `0` | Enable audio debug output |

## Usage

1. Run `python main.py`
2. Speak when prompted
3. The assistant will respond with voice
4. Say "exit", "quit", "goodbye", or "stop" to end

## Dependencies

Core dependencies are automatically installed with `pip install -r requirements.txt`. Some TTS backends may require additional system dependencies.