# rag_system.py

from typing import Iterator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

from config import CHROMA_DB_DIR, EMBED_MODEL, GROQ_API_KEY, GROQ_MODEL, SYSTEM_PROMPT, CHUNK_SIZE
from app_state import app_state

# ============ EMBEDDINGS + DB SETUP ============
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ Vector database loaded successfully")
except Exception as e:
    print(f"⚠️  Vector database setup failed: {e}")
    retriever = None

# ============ LLM SETUP ============
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY env var is required. Put it in .env")

try:
    chat_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.7)
    print("✅ Groq LLM initialized")
except Exception as e:
    print(f"❌ Groq initialization failed: {e}")
    raise

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