import os
import fitz
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
BM25_INDEX_DIR = "bm25_index"
PDF_DIR = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

def clean_text(text):
    return " ".join(text.replace("\n", " ").split())

def extract_text_with_metadata(pdf_dir):
    documents = []

    for filename in os.listdir(pdf_dir):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(pdf_dir, filename)
        pdf = fitz.open(file_path)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if not text.strip():
                continue

            documents.append({
                "source": filename,
                "page": page_num,
                "text": clean_text(text)
            })

    return documents

def split_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks_with_metadata = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": {
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_id": f"{doc['source']}_p{doc['page']}_c{i}"
                }
            })

    return chunks_with_metadata

def save_bm25_index(texts, metadatas, bm25_dir):
    os.makedirs(bm25_dir, exist_ok=True)
    with open(os.path.join(bm25_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)

    with open(os.path.join(bm25_dir, "metadatas.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)

def generate_vector_store(chunks, faiss_dir, bm25_dir):
    texts = [item["text"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    
    if not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir)
        
    vector_store.save_local(faiss_dir)
    print(f"Saved vector store to {faiss_dir}")

    save_bm25_index(texts, metadatas, bm25_dir)
    print(f"Saved BM25 index to {bm25_dir}")

def process_data(pdf_dir, index_dir, bm25_dir):
    print("Extracting text...")
    docs = extract_text_with_metadata(pdf_dir)

    print(f"Total pages processed: {len(docs)}")

    print("Splitting into chunks...")
    chunks = split_chunks(docs)

    print(f"Total chunks created: {len(chunks)}")

    print("Generating and saving vector store...")
    return generate_vector_store(chunks, index_dir, bm25_dir)

# CLI Support

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FAISS and BM25 indexes from PDFs")
    parser.add_argument("--pdf_dir", type=str, default="data", help="Directory with PDF files")
    parser.add_argument("--faiss_dir", type=str, default="faiss_index", help="Directory to save FAISS index")
    parser.add_argument("--bm25_dir", type=str, default="bm25_index", help="Directory to save BM25 index")

    args = parser.parse_args()
    process_data(pdf_dir=args.pdf_dir, index_dir=args.faiss_dir, bm25_dir=args.bm25_dir)

