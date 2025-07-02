import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
PDF_DIR = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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

def generate_vector_store(chunks, index_dir):
    texts = [item["text"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        
    vector_store.save_local(index_dir)
    print(f"Saved vector store to {index_dir}")

def process_data(pdf_dir, index_dir):
    print("Extracting text...")
    docs = extract_text_with_metadata(pdf_dir)

    print(f"Total pages processed: {len(docs)}")

    print("Splitting into chunks...")
    chunks = split_chunks(docs)

    print(f"Total chunks created: {len(chunks)}")

    print("Generating and saving vector store...")
    generate_vector_store(chunks, index_dir)

