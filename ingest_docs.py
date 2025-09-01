# ingest_docs.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

CHROMA_DB_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "./docs"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_docs = []

for file_name in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file_name)

    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_name.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    else:
        print(f"Skipping unsupported file: {file_name}")
        continue

    chunks = text_splitter.split_documents(docs)
    all_docs.extend(chunks)

# ============ STORE IN CHROMA ============
if all_docs:
    vectorstore.add_documents(all_docs)
    # Note: persist() is no longer needed in newer versions of Chroma
    print(f"✅ Ingested {len(all_docs)} chunks into Chroma DB")
else:
    print("⚠️ No documents found to ingest.")