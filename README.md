# 🧾 Talk to Your PDF

Talk to Your PDF is a lightweight AI-powered application that allows you to **ask questions and have conversations with the content of any PDF file**. It uses cutting-edge tools like **LLaMA 3**, **LangChain**, and **Hugging Face Hub** to extract insights and provide contextual answers from your documents.

---

## 🚀 Features

- 🧠 Powered by Meta’s **LLaMA 3** via Hugging Face Hub
- 📄 Upload any PDF file and interact with it using natural language
- 🔎 Context-aware question-answering using **vector-based semantic search**
- 🗂️ Document chunking & storage using **Chroma** and **LangChain’s text splitters**

---

## 🛠️ How It Works

1. **PDF Processing**
   - The uploaded PDF is loaded using `PyPDFLoader`.
   - The content is chunked using `RecursiveCharacterTextSplitter` to maintain semantic boundaries.
   - Embeddings for the chunks are generated using a custom embedding class and stored in a **Chroma** vector database.

2. **Conversational Retrieval**
   - A `ConversationalRetrievalChain` is created to retrieve relevant chunks from the vector store.
   - The chunks are passed as context to the LLM, which generates a coherent response to the user’s query.

---

## ⚙️ Installation (Local Setup)

Follow these steps to run the app locally:

```bash

# 1. Clone the repository
git clone https://github.com/yourusername/talk-to-your-pdf.git
cd talk-to-your-pdf

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # or use venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

```

## 🔐 Hugging Face API Token
You’ll need a free Hugging Face API token to run this project.

1. Go to: https://huggingface.co/settings/tokens

2. Generate a token and copy it.

3. Paste the token in your code (replace the placeholder at line 21 or use an .env file for better security).

## ▶️ Run the App
```bash

flask run

```
Then, open your browser and navigate to http://127.0.0.1:5000

## 📌 Requirements

- Python 3.8+
- Flask
- LangChain
- HuggingFaceHub
- PyPDFLoader
- Chroma
- Any LLaMA 3-compatible model hosted on Hugging Face

## ⚠️ Notes
Make sure you have a valid Hugging Face token before starting the app.

