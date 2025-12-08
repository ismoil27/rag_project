# # app/rag.py

# from app.retriever import search_mongodb
# from app.llm import generate_response

# def build_prompt(question: str, chunks: list):
#     context = ""

#     for c in chunks:
#         context += f"\n[FILE: {c['file']} | SCORE: {c['score']:.3f}]\n"
#         context += c["text"] + "\n"

#     prompt = f"""
# You are an assistant who answers based on the provided context.
# ONLY use the context. If the answer is not in the context, say: 'No relevant information found.'

# CONTEXT:
# {context}

# QUESTION:
# {question}
# """

#     return prompt


# async def ask_rag(question: str):
#     results = search_mongodb(question)

#     if len(results) == 0:
#         return "No related information found in the knowledge base."

#     prompt = build_prompt(question, results)

#     answer = await generate_response(prompt)

#     return {
#         "answer": answer,
#         "sources": results
#     }


# app/rag.py

from app.retriever import search_chroma
from app.llm import generate_response

def build_prompt(question: str, chunks: list):
    context = ""
    for c in chunks:
        context += f"\n--- FILE: {c['file']} (DISTANCE: {c['distance']:.4f}) ---\n"
        context += c["text"] + "\n"

    prompt = f"""
You are an AI assistant with access to parts of a codebase.
Use the provided code context below to answer the question.
Explain the answer in simple, clear terms.

If the answer is not fully obvious, infer it logically from the code.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (summarize based on the code above):
"""

    return prompt



async def ask_rag(question: str):
    retrieved_chunks = search_chroma(question)

    if len(retrieved_chunks) == 0:
        return {"answer": "No relevant information found."}

    prompt = build_prompt(question, retrieved_chunks)

    answer = await generate_response(prompt)

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }
