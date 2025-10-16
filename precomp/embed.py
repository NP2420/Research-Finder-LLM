import gzip
import json
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

'''
primary contains a bigger faiss file, including content such as:
    id, title, abstract, publish date, content type, ithaka doi, discipline names

    And is used as the fine in the coarse and fine setup.

secondary contains a generalized faiss file, including content such as:
    id, title, abstract

    And is used for the coarse in the coarse and fine setup.
'''


FILE = os.path.join("./resources", os.listdir("./resources")[0])
BATCH_SIZE = 50000          # process in chunks of 50k. Adjust depending on setup
SEGMENT_LIMIT = 5_000_000   # 5 million per big FAISS file to avoid memory issues. Adjust depending on setup.
FAISS_PERSIST_DIR = "./out/jstor"

print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)
print("Embeddings ready.")

def stream_json_records(file_path, type="primary"):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            if type == "primary":
                disciplines = [str(d) for d in (data.get('discipline_names') or [])]
                text = f"{data.get('title','')} {data.get('abstract','')} {' '.join(disciplines)}".strip()
                if text:
                    yield Document(
                        page_content=text,
                        metadata={
                            "id": data.get("item_id"),
                            "title": data.get("title", ""),
                            "abstract": data.get("abstract", ""),
                            "published_date": data.get("published_date"),
                            "content_type": data.get("content_type"),
                            "ithaka_doi": data.get("ithaka_doi"),
                            "discipline_names": data.get("discipline_names", [])
                        }
                    )
            elif type == "secondary":
                item_id = data.get("item_id")
                title = data.get("title", "")
                abstract = data.get("abstract", "")
                
                text = f"{title} {abstract}".strip()            
                if not text:
                    continue
                
                yield Document(
                    page_content=text,
                    metadata={"id": item_id}
                )
            else:
                raise ValueError(f"{type} is not an accepted value. Try 'primary' or 'secondary'.")
                
def ingest_records(file_path, type="primary"):
    print(f"Starting ingestion from {file_path}...")

    buffer = []
    total_docs = 0
    segment_num = 1
    batch_num = 1
    vectorstore = None

    for doc in stream_json_records(file_path, type): #per line
        buffer.append(doc)
        total_docs += 1

        if len(buffer) >= BATCH_SIZE:
            print(f"Processing batch {batch_num}: ({total_docs - len(buffer) + 1}-{total_docs})")
            new_store = FAISS.from_documents(buffer, embeddings)
            buffer = []

            if vectorstore is None:
                vectorstore = new_store
            else:
                vectorstore.merge_from(new_store)

            batch_num += 1

        # Save segment if we hit 5 million documents
        if total_docs % SEGMENT_LIMIT == 0:
            segment_path = os.path.join(FAISS_PERSIST_DIR + "_" + type, f"segment_{segment_num}")
            print(f"Saving segment {segment_num} ({total_docs:,} total docs) to {segment_path}")
            os.makedirs(segment_path, exist_ok=True)
            vectorstore.save_local(segment_path)
            vectorstore = None
            segment_num += 1
            batch_num = 1

    # Save final segment
    if buffer:
        print(f"Processing final {len(buffer)} docs...")
        new_store = FAISS.from_documents(buffer, embeddings)
        if vectorstore is None:
            vectorstore = new_store
        else:
            vectorstore.merge_from(new_store)

    if vectorstore:
        segment_path = os.path.join(FAISS_PERSIST_DIR + "_" + type, f"segment_{segment_num}")
        print(f"Saving final segment {segment_num} at {segment_path}")
        os.makedirs(segment_path, exist_ok=True)
        vectorstore.save_local(segment_path)

    print("Ingestion complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_records(FILE, "primary")
    ingest_records(FILE, "secondary")