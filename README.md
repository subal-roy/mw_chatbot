# 🤖 MW Chatbot – AI-Powered HR Assistant

MW Chatbot is an internal HR assistant built with **Streamlit**, **LangChain**, **Gemini (Google Generative AI)**, **Cohere**, and a **hybrid RAG pipeline** using FAISS, BM25, and Google CSE fallback.

It helps employees get instant answers about company policies, holidays, HR procedures, and more by intelligently retrieving and reranking information from internal PDFs and external sources.

---

## 🚀 Key Features
### 🔍 Hybrid Retrieval System  
Combines:
- **Dense retrieval** using FAISS
- **Sparse retrieval** using BM25
- **Google CSE fallback** for external web search when internal docs lack coverage

### 🔁 Reranking with Cohere  
All candidate documents—internal or external—are reranked using **Cohere's Rerank API (v3.0)** to ensure the most relevant context is used.

### 🧠 RAG-Based Question Answering
Utilizes Retrieval-Augmented Generation with Gemini (Google Generative AI) for accurate, contextual responses.

### 📄 Intelligent PDF Parsing
Automatically extracts and chunks text from HR-related PDFs

### ⚡ Optimized Retrieval Performance
Uses a singleton pattern to load retrievers and vector stores only once, minimizing resource usage.

### 💬 Streamlit Chat Interface
Interactive and user-friendly web UI for employees to ask questions about HR policies, leave, and more.

## ⚙️ Installation (Local)

### 1. Clone the repo

```bash
git clone https://github.com/subal-roy/mw_chatbot.git
cd mw_chatbot
````

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your `.env` file

Create a `.env` file with your Gemini API key:

```
GOOGLE_API_KEY="Google api key"
SEARCH_ENGINE_ID="google CSE search engine id"
COHERE_API_KEY="cohere api key"
```

---

## 🏗️ Build Indexes
🗂️ Make sure your PDF documents are placed inside the data/ directory.
```bash
# Windows
build_index.bat

# Or manually
python data_processor.py --pdf_dir=data --faiss_dir=faiss_index --bm25_dir=bm25_index
```

---

## 🧠 Run the Chatbot

```bash
streamlit run app.py
```

Access it at [http://localhost:8501](http://localhost:8501)