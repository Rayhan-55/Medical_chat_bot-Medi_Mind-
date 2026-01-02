# 🩺 MediMind AI
### AI-powered Medical Chatbot & Skin Disease Detection System

MediMind AI is a **Flask-based AI healthcare assistant**, combining a medical chatbot with skin disease detection.

---

## 🚀 Features

### 🤖 Medical Chatbot
- Uses **Retrieval-Augmented Generation (RAG)**  
- Vector search powered by **Pinecone**  
- LLM powered by **Groq (LLaMA 3.1)**  
- Supports **English & Bengali**  
- Maintains chat memory per session  

### 🧴 Skin Disease Detection
- Detects 10 common skin diseases using a **CNN model**  
- Real-time live camera detection  
- Image upload-based detection  
- Provides confidence scores for predictions  

### 🌐 Web Interface
- Modern UI using HTML, CSS, and JavaScript  
- Separate pages for Chat & Live Skin Detection  

---

## 🧠 Supported Skin Diseases

- Eczema  
- Melanoma  
- Atopic Dermatitis  
- Basal Cell Carcinoma  
- Melanocytic Nevi (NV)  
- Benign Keratosis  
- Psoriasis  
- Seborrheic Keratoses  
- Tinea / Ringworm / Candidiasis  
- Warts / Molluscum  

---


---

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/Rayhan-55/Medical_chat_Bot.git
---
2.Navigate to project folder:
cd MediMind
--
3.Create virtual environment:
python -m venv venv
--
4.Activate environment:
Windows:
venv\Scripts\activate
Linux/Mac:
source venv/bin/activate
----
5.Install dependencies:
pip install -r requirements.txt
-----
6.Run the app:
python app.py

