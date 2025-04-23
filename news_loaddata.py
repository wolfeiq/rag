import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="eu-central-1",
    #    model_id= "amazon.titan-text-express-v1"
    )
    

    return embeddings

FAISS_PATH = ""
DATA_PATH = ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_faiss(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_faiss(chunks: list[Document]):
    chunks_with_ids = calculate_chunk_ids(chunks)
    if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
        db = FAISS.load_local(FAISS_PATH, get_embedding_function())
        existing_ids = set([metadata["id"] for metadata in db.docstore._dict.values()])
        print(f"Number of existing documents in DB: {len(existing_ids)}")
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
                
        if len(new_chunks):
            new_db = FAISS.from_documents(new_chunks, get_embedding_function())
            db.merge_from(new_db)
            db.save_local(FAISS_PATH)
        else:
            print(":( nothing new to add")
    else:
        os.makedirs(FAISS_PATH, exist_ok=True)
        db = FAISS.from_documents(chunks_with_ids, get_embedding_function())
        db.save_local(FAISS_PATH)
        print(f"Added {len(chunks_with_ids)} documents")
        
def calculate_chunk_ids(chunks):
    #  "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

if __name__ == "__main__":
    main()
