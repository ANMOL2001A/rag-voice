
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_DB_DIR, EMBED_MODEL


DATA_DIR = "docs"

def ingest_documents():
    """
    Loads documents from the data directory, splits them into chunks,
    and stores them in a Chroma vector database.
    """
    print("Starting document ingestion...")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    except Exception as e:
        print(f"❌ Error initializing embeddings or Chroma: {e}")
        return

    # 2. Define text splitters
    # Generic splitter for fallback
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Specialized Markdown splitter
    headers_to_split_on = [
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    all_chunks = []
    processed_files = 0

    # 3. Load and process files from the data directory
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        
        try:
            if file_name.endswith(".pdf"):
                print(f"Processing PDF: {file_name}")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                chunks = recursive_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                processed_files += 1

            elif file_name.endswith(".md"):
                print(f"Processing Markdown: {file_name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                chunks = markdown_splitter.split_text(md_content)
             
                all_chunks.extend(chunks)
                processed_files += 1

            elif file_name.endswith(".txt"):
                print(f"Processing Text: {file_name}")
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                chunks = recursive_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                processed_files += 1
            
            elif file_name.endswith(".docx"):
                print(f"Processing DOCX: {file_name}")
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                chunks = recursive_splitter.split_documents(docs)
                all_chunks.extend(chunks)
                processed_files += 1
                
            else:
                print(f"🟡 Skipping unsupported file type: {file_name}")
                continue

        except Exception as e:
            print(f"❌ Error processing file {file_name}: {e}")

    # 4. Store chunks in Chroma DB
    if all_chunks:
        print(f"Adding {len(all_chunks)} new document chunks to the vector store...")
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingestion complete. Processed {processed_files} files.")
    else:
        print("⚠️ No new documents were found to ingest.")

if __name__ == "__main__":
    ingest_documents()
