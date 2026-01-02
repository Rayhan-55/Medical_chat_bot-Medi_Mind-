from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import traceback
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from skin_modle import predict_skin_disease
from PIL import Image
from io import BytesIO
import base64

load_dotenv()
app = Flask(__name__)

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


try:
    print(f"Pinecone Key Status: {'Loaded' if os.getenv('PINECONE_API_KEY') else 'MISSING'}")
    print(f"Groq Key Status: {'Loaded' if os.getenv('GROQ_API_KEY') else 'MISSING'}")
    
   
    original_embeddings = download_hugging_face_embeddings()

    class SafeEmbeddings:
        def embed_query(self, text):
            return original_embeddings.embed_query(str(text).replace("\n", " ").strip())
        def embed_documents(self, texts):
            return original_embeddings.embed_documents([str(t).replace("\n", " ").strip() for t in texts])

    embeddings = SafeEmbeddings()

  
    vectorstore = PineconeVectorStore(index_name="medichabot", embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

   
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4, max_tokens=1024)
    
    print("Core RAG Components Initialized Successfully.")

except Exception as e:
    print("\n!!!!!!!!!!!!!!! CRITICAL STARTUP ERROR !!!!!!!!!!!!!!!")
    print(f"ERROR DURING RAG SETUP: {type(e).__name__}: {e}")
    error = traceback.format_exc()
    print("ERROR OCCURRED:\n", error)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    raise e 



prompt = ChatPromptTemplate.from_messages([
    ("system", "You are MediMind AI, a helpful medical assistant. Answer in Bengali if asked in Bengali, otherwise in English. Be concise."),
    ("system", "Use the following context to answer the user's question:\n\n{context}"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""


to_string = RunnableLambda(lambda x: str(x.content if hasattr(x, "content") else x))


rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"])), 
    )
    | prompt
    | llm
    | to_string
)

session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("msg", "").strip()
        if not user_input:
            return "Please type something."
        
        print("--- STARTING RAG CHAIN INVOCATION ---") 
        
        result = chain_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "user123"}}
        )

       
        print(f"User: {user_input}") 
        print(f"Bot: {result}")
        print("-" * 70)

        return result

    except Exception as e:
        
        print("\n!!!!!!!!!!!!!!! TRACEBACK CAPTURED (IN CHAT ROUTE) !!!!!!!!!!!!!!!")
        print(f"EXCEPTION OBJECT: {type(e).__name__}: {e}")
        error = traceback.format_exc()
        print("ERROR OCCURRED:\n", error) 
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return "Sorry, server is busy. Try again."

@app.route("/realtime")
def realtime():
    return render_template("realtime.html")

@app.route("/realtime_detect", methods=["POST"])
def realtime_detect():
   
    try:
        
        image_data_uri = request.json['image']
        
        
        if 'base64,' in image_data_uri:
            encoded_data = image_data_uri.split('base64,')[1]
        else:
            
            encoded_data = image_data_uri
        
        
        img_bytes = base64.b64decode(encoded_data)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        
        disease, confidence = predict_skin_disease(img)
        
       
        return jsonify({"disease": disease, "confidence": round(float(confidence), 1)})
        
    except Exception as e:
        
        print(f"!!! REALTIME DETECT EXCEPTION: {type(e).__name__}: {e}") 
        
        return jsonify({"disease": "Error", "confidence": 0})

@app.route("/upload_detect", methods=["POST"])
def upload_detect():
    
    try:
        img = Image.open(request.files['image'].stream).convert("RGB")
        disease, confidence = predict_skin_disease(img)
        
        return jsonify({"disease": disease, "confidence": round(float(confidence), 1)})
    except Exception as e:
        print(f"!!! UPLOAD DETECT EXCEPTION: {type(e).__name__}: {e}") 
        return jsonify({"disease": "Error", "confidence": 0})


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MediMind AI STARTED SUCCESSFULLY!")
    print("Open in browser: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)