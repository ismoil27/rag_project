# app/vectorstore.py

import chromadb

# NEW PERSISTENT CLIENT (correct for Chroma 0.5+)
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Create / get collection
collection = chroma_client.get_or_create_collection(
    name="burak_rag",
    metadata={"hnsw:space": "cosine"}  # metric
)
