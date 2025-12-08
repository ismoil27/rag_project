# app/llm.py
import ollama

async def generate_response(prompt: str) -> str:
    """
    Calls the local Llama model via Ollama.
    """
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
