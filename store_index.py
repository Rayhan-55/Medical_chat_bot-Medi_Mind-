import time
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec  
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "medical-chatbot"


existing_indexes = [index_info["name"] for index_info in pc.list_indexes().names()]
print(f"Existing indexes: {existing_indexes}")

if index_name not in existing_indexes:
    print(f"Creating index '{index_name}'...")
    
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  
    )
    
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")


print("Loading PDFs from data/ folder...")
loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents.")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


print("Uploading to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)
print("Upload complete! Index ready for querying.")


query = "What is diabetes?"
result = docsearch.similarity_search(query, k=2)
print("Test query result:")
for doc in result:
    print(f"- {doc.page_content[:100]}...")