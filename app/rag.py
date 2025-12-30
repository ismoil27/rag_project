from app.retriever import search_chroma
from app.llm import generate_response

def build_prompt(question: str, chunks: list):
    context = ""
    for c in chunks:
        context += f"\n--- FILE: {c['file']} (DISTANCE: {c['distance']:.4f}) ---\n"
        context += c["text"] + "\n"

    prompt = f"""
You are an AI assistant analyzing a backend codebase.

IMPORTANT RULES:
- If a function or method name appears in the question, look carefully for that
  exact function inside the provided context.
- Do NOT say a function does not exist unless you are sure it is not present
  in the context.
- If the exact name is not found, look for the closest matching function
  (same controller or same file) and explain that instead.
- Do NOT hallucinate code that is not shown.

Use the code below as the only source of truth.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
- Be short
- Be concrete
- Explain what the function does
- Mention the file if helpful
"""
    return prompt



def build_debug_prompt(question: str, chunks: list):
    context = ""
    for c in chunks:
        context += f"\n--- FILE: {c['file']} ---\n{c['text']}\n"

    prompt = f"""
You are a senior backend engineer helping a teammate debug an issue.

Goal:
- Identify the real cause of the error
- Explain it clearly and respectfully
- Focus on the fix, not on blame

How to answer:
- Do NOT say "user code" or "repository code"
- Do NOT sound accusatory
- Speak like a helpful teammate
- Be direct and practical

What to do:
1. Look at the code patterns shown below.
2. Compare them with the code in the question.
3. Point out what is missing or different (e.g. async / await).
4. Explain why that causes a runtime or logic error.
5. Show the corrected version of the code.

REFERENCE CODE PATTERNS:
{context}

QUESTION:
{question}

ANSWER (clear, friendly debugging explanation):
"""
    return prompt


async def ask_rag(question: str):
    retrieved_chunks = search_chroma(question)

    if not retrieved_chunks:
        return {"answer": "No relevant information found."}

    if is_debug_question(question):
        prompt = build_debug_prompt(question, retrieved_chunks)
    else:
        prompt = build_prompt(question, retrieved_chunks)

    answer = await generate_response(prompt)

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }




def is_debug_question(question: str) -> bool:
    keywords = [
        "error",
        "why",
        "not working",
        "gives me error",
        "bug",
        "issue",
        "wrong",
        "fix"
    ]
    q = question.lower()
    return any(k in q for k in keywords)
