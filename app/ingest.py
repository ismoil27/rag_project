# app/ingest.py
import os
import glob
import ollama
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.vectorstore import collection

# Load files
def load_files(folder: str):
    data = []
    for filepath in glob.glob(f"{folder}/**/*", recursive=True):
        if os.path.isfile(filepath) and filepath.endswith((".py", ".js", ".ts", ".md", ".txt")):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    data.append({"file": filepath, "text": text})
            except:
                pass
    return data


# Chunk text
def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_text(text)


# Generate embedding using Ollama nomic-embed-text
def embed_text(text: str):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]


def ingest_project(folder: str):
    print(f"Loading files from: {folder}")
    files = load_files(folder)
    print(f"Found {len(files)} files.")

    for file_data in files:
        chunks = chunk_text(file_data["text"])

        print(f"File: {file_data['file']} — Chunks: {len(chunks)}")

        for chunk in chunks:
            emb = embed_text(chunk)

            doc_id = str(uuid.uuid4())  # unique ID

            collection.add(
                ids=[doc_id],
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"file": file_data["file"]}]
            )

    print("Ingestion complete!")
    print("Chroma DB stored at: chroma_db/")
