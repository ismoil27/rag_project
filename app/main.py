# app/main.py
from fastapi import FastAPI
from app.llm import generate_response
from app.ingest import ingest_project

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG project is running"}

@app.post("/chat")
async def chat(input: dict):
    prompt = input.get("prompt")
    answer = await generate_response(prompt)
    return {"answer": answer}




@app.post("/ingest")
async def ingest(data: dict):
    folder = data.get("folder")
    ingest_project(folder)
    return {"status": "ok", "folder": folder}



from app.rag import ask_rag

# @app.post("/ask")
# async def ask(data: dict):
#     question = data.get("question")
#     response = await ask_rag(question)
#     return response

from app.rag import ask_rag

@app.post("/ask")
async def ask(data: dict):
    question = data.get("question")
    response = await ask_rag(question)
    return response
