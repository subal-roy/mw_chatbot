import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
import os

# Globals to hold singletons
_embeddings = None
_hybrid_retriever = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    return _embeddings
  

def get_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        embeddings = get_embeddings()
        faiss = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
        faiss_retriever = faiss.as_retriever(search_kwargs={"k":10})

        with open(os.path.join("bm25_index", "texts.json"), "r", encoding="utf-8") as f:
                texts = json.load(f)

        with open(os.path.join("bm25_index", "metadatas.json"), "r", encoding="utf-8") as f:
            metadatas = json.load(f)
        
        bm25_retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
        bm25_retriever.k = 10

        _hybrid_retriever = EnsembleRetriever(
            retrievers = [faiss_retriever, bm25_retriever],
            weights = [0.5, 0.5]
        )
    return _hybrid_retriever