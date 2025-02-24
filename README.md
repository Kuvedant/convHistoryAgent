# AI Chat Agent with RAG 🚀

## **📌 Overview**
This project implements an **AI-powered Chat Agent** using **Retrieval-Augmented Generation (RAG)**. It leverages:
- **FastAPI** for the backend
- **FAISS** for efficient similarity search
- **LangChain** for conversational AI
- **Guardrails AI** for content moderation
- **Streamlit** for the frontend UI
- **Docker** for easy deployment

The AI system can **retrieve relevant context, classify conversations, and verify responses** while ensuring **safe content** using Guardrails.

---

## **📂 Project Structure**
```
.
├── backend.py             # FastAPI backend with RAG & Guardrails
├── frontend.py            # Streamlit UI for chat interface
├── requirements.txt       # Dependencies list
├── Dockerfile             # Docker configuration
├── text_1.json           # Sample conversation data
├── README.md              # Documentation
```
---

## **🛠 Installation & Setup**
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Run the Backend (FastAPI)**
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```
**Test API:** Open `http://127.0.0.1:8000/docs` in the browser.

### **3️⃣ Run the Frontend (Streamlit)**
```bash
streamlit run frontend.py
```

### **4️⃣ Test API with `curl`**
```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"question": "Hello"}'
```

---

## **🛡 Guardrails AI Integration**
This project integrates **Guardrails AI** to filter **toxic content** in:
- 📜 **Retrieved Context**
- 📌 **Classification**
- ✅ **Verification Score**

### **Custom Toxicity Validation**
Implemented **custom validators** to block inappropriate responses using **LLM-based toxicity filtering** & keyword blocking.

---

## **📦 Docker Deployment**
### **1️⃣ Build Docker Image**
```bash
docker build -t ai-chat-agent .
```

### **2️⃣ Run the Container**
```bash
docker run -p 8000:8000 ai-chat-agent
```

### **3️⃣ Access the API**
Visit `http://127.0.0.1:8000/docs` for Swagger UI.

---

## **🔗 API Endpoints**
### **1️⃣ Query API**
- **Endpoint:** `POST /query`
- **Request:**
  ```json
  {
    "question": "What is AI?"
  }
  ```
- **Response:**
  ```json
  {
    "question": "What is AI?",
    "answer": "Artificial Intelligence (AI) refers to...",
    "retrieved_context": ["AI is...", "Machine Learning..."],
    "classification": "Work",
    "verification_score": 0.98
  }
  ```

### **2️⃣ Home API**
- **Endpoint:** `GET /`
- **Response:** `{ "message": "API is running! Use /docs for Swagger UI" }`

---

## **🛠 Technologies Used**
- **Python** (FastAPI, LangChain, FAISS, Streamlit)
- **OpenAI GPT** (LLM for response generation)
- **Guardrails AI** (Content filtering & moderation)
- **Docker** (Containerized deployment)
- **FAISS** (Efficient retrieval of past messages)
- **Sentence Transformers** (Embedding generation)

---

## **🚀 Future Improvements**
- ✅ Improve **response accuracy** with fine-tuned LLMs
- ✅ Deploy on **cloud (AWS/GCP/Azure)** for scalability
- ✅ Add **user authentication** for secure access

---

## **📧 Contact & Support**
For issues or feature requests, create an **Issue** or **Pull Request** in the repository.

🚀 “AI agents will transform the way we interact with technology, making it more natural and intuitive. They will enable us to have more meaningful and productive interactions with computers.” ~Fei-Fei Li
🎯

