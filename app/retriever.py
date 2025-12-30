# app/retriever.py
import ollama
from app.vectorstore import collection

def embed_query(text: str):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]


def search_chroma(query: str, top_k: int = 20):
    q_emb = embed_query(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    # Transform Chroma output
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "file": results["metadatas"][0][i]["file"],
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i]
        })

    return docs
