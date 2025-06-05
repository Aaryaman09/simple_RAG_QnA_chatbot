# 🧠 RAG Chatbot using LangChain, Ollama, and Chroma

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot built with [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), and [Chroma](https://www.trychroma.com/) as the vector database. It supports multi-turn conversations with session-based memory and provides intelligent responses grounded on your custom documents.

---

## 🔧 Tech Stack

- **LangChain** – Framework to build LLM-powered applications
- **Ollama** – Local LLM inference (e.g. LLaMA3.2)
- **Chroma** – Embedded document vector store for retrieval
- **HGF Embeddings** – Huggingface embeddings by model : all-MiniLM-L6-v2
- **Python 3.12+**

---

## 🚀 How It Works
1. Document Loading – Source data (text, Website, etc.) is loaded and split into chunks.
2. Embedding & Storage – Chunks are embedded and stored in Chroma vector DB.
3. Prompting – A custom prompt is fetched to guide the LLM response.
4. RAG Chain Creation – Combines retriever, prompt, and Ollama-backed LLM.
5. Session Memory – Tracks user history across a session with RunnableWithMessageHistory.

---

🛠️ Setup Instructions
1. Clone the Repo
```bash
    git clone https://github.com/Aaryaman09/simple_RAG_QnA_chatbot.git
    cd simple_RAG_QnA_chatbot
```

2. Install Dependencies
```bash
    pip install -r requirements.txt
```

### Make sure you have Ollama installed and running locally:
* Change model as per your desire but do update the key.json

```bash
    ollama run llama3.2 
```

3. Run the Chatbot
```bash
    python app.py
```

---

## Screenshots

1. RAG application with Session ID

<img src="https://raw.githubusercontent.com/Aaryaman09/simple_RAG_QnA_chatbot/refs/heads/main/screenshot/RAG_with_Session_Id.png" alt="RAG Architecture" width="800"/>

### **Note** : Check readme in screenshot folder to know whats happening in this picture.
