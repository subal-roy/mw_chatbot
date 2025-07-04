# 🤖 MW Chatbot – AI-Powered HR Assistant

MW Chatbot is an internal HR assistant built with **Streamlit**, **LangChain**, **Gemini (Google Generative AI)**, and a **hybrid RAG pipeline** using FAISS and BM25.

It can answer employee queries about HR policies, holidays, and other internal topics by retrieving relevant information from PDF documents.

---

## 🚀 Key Features
#### 🔍 Hybrid Retrieval System
Combines dense (FAISS) and sparse (BM25) search methods for highly relevant and balanced results.

#### 🧠 RAG-Based Question Answering
Utilizes Retrieval-Augmented Generation with Gemini (Google Generative AI) for accurate, contextual responses.

#### 📄 Intelligent PDF Parsing
Automatically extracts and chunks text from HR-related PDFs

#### ⚡ Optimized Retrieval Performance
Uses a singleton pattern to load retrievers and vector stores only once, minimizing resource usage.

#### 💬 Streamlit Chat Interface
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
GOOGLE_API_KEY=your_google_genai_api_key
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