# create_vector_db.py ← এভাবে রাখো (ইংরেজি প্রিন্ট)

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv          # <-- এই লাইনটা যোগ করেছি
load_dotenv() 

print("Loading PDFs...")
documents = load_pdf_file("data/")
print(f"Loaded: {len(documents)} PDFs")

print("Chunking text...")
chunks = text_split(documents)
print(f"Created: {len(chunks)} chunks")

print("Loading embeddings...")
embeddings = download_hugging_face_embeddings()

print("Uploading to Pinecone... (may take 1-3 mins)")
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="medichabot"        # তোমার আসল ইনডেক্স নাম
)

print("\nAll done!")
print("Now run → python app.py")