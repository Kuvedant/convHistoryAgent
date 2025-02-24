import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import openai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
import os

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI
app = FastAPI()

# Add CORS Middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# API Models
class QueryRequest(BaseModel):
    question: str

class ResponseModel(BaseModel):
    question: str
    answer: str
    retrieved_context: list
    classification: str
    verification_score: float

# Load conversation data with error handling
try:
    with open("text_1.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    conversations = data.get("mess", [])
except Exception as e:
    raise RuntimeError(f"Error loading conversation data: {e}")

# Initialize sentence transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare messages for embedding
messages = [entry.get("user", "") for entry in conversations if "user" in entry]
message_embeddings = embedding_model.encode(messages)

# Create FAISS index for vector storage
vector_dim = message_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(vector_dim)
faiss_index.add(np.array(message_embeddings))

# Store message mapping
message_mapping = {i: messages[i] for i in range(len(messages))}

def retrieve_context(query, top_k=3):
    try:
        query_embedding = embedding_model.encode([query])
        _, indices = faiss_index.search(np.array(query_embedding), top_k)
        retrieved_texts = [message_mapping[idx] for idx in indices[0] if idx in message_mapping]
        return retrieved_texts
    except Exception as e:
        return [f"Error in retrieval: {str(e)}"]


def classify_conversation(text):
    categories = ["Casual Chat", "Hobbies", "Work", "Personal Matters"]
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Classify into one of these categories: {', '.join(categories)}"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in classification: {str(e)}"

def generate_response(question, context):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Use this context to answer: {context}"},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def verify_answer(answer, context):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Does this answer '{answer}' match the context '{context}'? Reply with a confidence score (0-1)."},
                {"role": "user", "content": "Give me a numerical confidence score only."}
            ]
        )
        return float(response.choices[0].message.content.strip())
    except Exception as e:
        return 0.0


@app.post("/query", response_model=ResponseModel)
def query_rag(request: QueryRequest):
    try:
        retrieved_context = retrieve_context(request.question)
        generated_answer = generate_response(request.question, retrieved_context)
        classification = classify_conversation(request.question)
        verification_score = verify_answer(generated_answer, retrieved_context)
        
        return ResponseModel(
            question=request.question,
            answer=generated_answer,
            retrieved_context=retrieved_context,
            classification=classification,
            verification_score=verification_score
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def home():
    return {"message": "API is running! Use /docs for Swagger UI"}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

