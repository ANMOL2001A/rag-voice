# Enhanced Voice and Text RAG Assistant

A voice-enabled and text-based RAG (Retrieval-Augmented Generation) assistant that allows natural conversation with your documents through speech or a command-line interface.

## Project Structure

```
/
├── main.py                    # Main entry point for VOICE assistant
├── query_terminal.py          # Entry point for TEXT-BASED querying
├── conversation_handler.py    # Voice conversation loop logic
├── rag_system.py              # RAG system with vector DB and LLM
├── ingest_docs.py             # Script to load documents into the DB
|
├── config.py                  # Configuration and environment variables
├── app_state.py               # Thread-safe application state management
├── audio_utils.py             # Audio processing, recording, and transcription
├── tts_engine.py              # Text-to-speech engine and workers
|
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── docs/                      # Directory for your source documents
└── chroma_db/                 # Default vector database location
```

## Setup Instructions

1.  **Clone the repository and navigate to the directory.**

2.  **Create a virtual environment:**
    ```bash
    python -m venv myvenv
    source myvenv/bin/activate 
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root directory by copying the example below. This file stores your API keys and configuration.
    ```
    GROQ_API_KEY="your_groq_api_key_here"
    ```

5.  **Place your documents** into the `docs/` directory. The `ingest_docs.py` script currently supports PDF files.

6.  **Ingest your documents** into the vector database. This command will process the files in the `docs/` folder and store them in the `chroma_db/` directory.
    ```bash
    python ingest_docs.py
    ```

## Application Flow

The application works in two main modes: Voice or Text.

### 1. Voice Assistant Flow (`main.py`)
-   **`main.py`**: The main entry point that initializes and starts all components.
-   **`conversation_handler.py`**: This script orchestrates the voice interaction.
    - It uses **`audio_utils.py`** to listen via the microphone, detect speech, and transcribe it to text using the Whisper model (via Groq).
-   **`rag_system.py`**: The user's transcribed query is sent here.
    - It searches the Chroma vector database for the 3 most relevant document chunks.
    - It combines the user's query and the retrieved context into a prompt.
    - This prompt is sent to a Large Language Model (LLM) using Groq (e.g., Llama 3) to generate a response.
-   **`tts_engine.py`**: The LLM's response text is converted back into speech in real-time.
-   The audio is played back to the user, completing the loop.

### 2. Text-Based Query Flow (`query_terminal.py`)
-   **`query_terminal.py`**: This script provides a simple command-line interface for interacting with your documents.
-   It takes a text query directly from the terminal.
-   The query is sent to the **`rag_system.py`**, which performs the same retrieval and generation process as in the voice flow (searching ChromaDB and calling the LLM).
-   The final response from the LLM is printed directly to the terminal.

## How to Run

You must have completed the setup instructions above, especially ingesting your documents.

### Running the Text-Based Query Terminal
This is the simplest way to interact with your documents.

```bash
python query_terminal.py
```
You will see a `>` prompt. Type your question and press Enter. Type `exit` or `quit` to end the session.

### Running the Voice Assistant
This provides a full voice-to-voice interaction.

```bash
python main.py
```
The application will calibrate your microphone's noise level and then prompt you to speak. Say "exit", "quit", or "goodbye" to stop the assistant.
