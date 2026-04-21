
#  langgraph_rag_app (LangGraph + Groq + Streamlit)

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF files and ask questions based on their content. It uses LangGraph for workflow management, Groq for fast LLM responses, and Streamlit for the user interface.

---

## Features
- Upload PDF documents
- Extract and process text from PDFs
- Split documents into chunks
- Generate embeddings for semantic search
- Store and retrieve relevant context
- Ask questions about uploaded documents
- Get fast responses using Groq LLM
- Simple web interface using Streamlit

---

## Tech Stack
- Python
- Streamlit (Frontend)
- LangGraph (Workflow orchestration)
- LangChain (RAG pipeline support)
- Groq API (LLM inference)
- FAISS / Chroma (Vector database - optional)
- PyPDF / PDFMiner (PDF processing)

---

## Project Structure
```

pdf-rag-assistant/
│
├── rag_frontend.py                  # Streamlit frontend
├── backend/
│   ├── rag_backend.py    
        
    
│
├── data/                  # Uploaded PDFs
├── requirements.txt
└── README.md

```

---

## Installation

### 1. Clone the repository
```

git clone [https://github.com/your-username/pdf-rag-assistant.git](https://github.com/your-username/pdf-rag-assistant.git)
cd pdf-rag-assistant

```

### 2. Create virtual environment
```

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

```

### 3. Install dependencies
```

pip install -r requirements.txt

```

---

## Setup Groq API Key
Create a `.env` file and add:

---

## Run the Application
```

streamlit run rag_frontend.py

```

---

## How It Works
1. User uploads a PDF file
2. Text is extracted and split into chunks
3. Embeddings are created and stored in a vector database
4. User asks a question
5. Relevant context is retrieved
6. Groq LLM generates an answer using retrieved context

---

#

