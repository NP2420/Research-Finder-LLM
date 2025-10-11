import gzip
import json
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

FILE = "./resources/jstor_metadata_2025-10-09.jsonl.gz"
BATCH_SIZE = 50000
FAISS_PERSIST_DIR = "../resources/jstor"

print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)
print("Embeddings ready.")

def stream_json_records(file_path, max_items=None):
    """
    Stream records from a .jsonl.gz file.
    Each yielded record is a dict with searchable text and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)

            # Combine title, abstract, and discipline names for embedding
            disciplines = data.get('discipline_names') or []
            disciplines = [str(d) for d in disciplines]
            text = f"{data.get('title','')} {data.get('abstract','')} {' '.join(disciplines)}".strip()

            if text:
                metadata = {
                    "id": data.get("item_id"),
                    "title": data.get("title", ""),
                    "abstract": data.get("abstract", ""),
                    "published_date": data.get("published_date"),
                    "content_type": data.get("content_type"),
                    "ithaka_doi": data.get("ithaka_doi"),
                    "discipline_names": data.get("discipline_names", [])
                }
                yield {"text": text, "metadata": metadata}

            if max_items and i >= max_items:
                break

def ingest_records(file_path, batch_size=BATCH_SIZE):
    """
    Ingest records into a single persistent FAISS vector store in batches.
    """
    print(f"Starting ingestion from {file_path}...")
    buffer = []
    batch_num = 1
    vectorstore = None

    for i, record in enumerate(stream_json_records(file_path, 100), 1):
        buffer.append(Document(page_content=record["text"], metadata=record["metadata"]))

        if i % batch_size == 0:
            print(f"Processing batch {batch_num}: ({i-batch_size+1}-{i})")
            new_store = FAISS.from_documents(buffer, embeddings)

            if vectorstore is None:
                vectorstore = new_store
            else:
                vectorstore.merge_from(new_store)

            buffer = []
            batch_num += 1

    # Process remaining records
    if buffer:
        print(f"Processing final batch ({i - len(buffer) + 1}-{i})...")
        new_store = FAISS.from_documents(buffer, embeddings)
        if vectorstore is None:
            vectorstore = new_store
        else:
            vectorstore.merge_from(new_store)

    # Save merged vector store
    if vectorstore:
        vectorstore.save_local(FAISS_PERSIST_DIR)
        print(f"Ingestion complete. FAISS vector store saved at: {FAISS_PERSIST_DIR}")
    else:
        print("No records were ingested.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_records(FILE)
