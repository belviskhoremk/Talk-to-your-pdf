# 🧾 Talk to Your PDF

Upload any PDF and **ask questions about it in plain language**. The app finds
the passages relevant to your question and has an LLM (Meta **Llama 3 8B
Instruct**, via the Hugging Face Inference API) answer from them. It also
remembers the conversation, so follow-up questions work.

Built with **Flask**, **LangChain**, **Sentence-Transformers** and **Chroma**.

## How it works

This is a classic **Retrieval-Augmented Generation (RAG)** pipeline:

```
            upload                                          ask
PDF ──► PyPDFLoader ──► text splitter ──► embeddings ──► Chroma DB
                         (1500 chars,     (all-MiniLM-L6-v2)   │
                          150 overlap)                         ▼
question + chat history ──► standalone question ──► top-k similar chunks
                                                              │
                                    Llama 3 ◄── prompt with chunks as context
                                       │
                                     answer
```

1. **Load**: the PDF is read page by page.
2. **Split**: pages are cut into overlapping ~1500-character chunks, so each
   one fits in the prompt and no sentence is lost at a boundary.
3. **Embed & store**: each chunk becomes a vector and goes into an in-memory
   Chroma database.
4. **Retrieve & answer**: a follow-up question is first rewritten into a
   standalone question using the chat history. The most similar chunks are
   then retrieved and passed to the LLM as context.

## Project structure

```
.
├── app.py            # Flask routes: upload, chat page, /ask JSON API
├── rag.py            # RAG pipeline: load, split, embed, retrieve, answer
├── config.py         # Settings read from environment variables / .env
├── templates/
│   ├── index.html    # Upload page
│   └── chatbot.html  # Chat UI
├── requirements.txt
└── .env.example      # Template for your local .env
```

## Getting started

**Requirements:** Python 3.10–3.12 and a free
[Hugging Face access token](https://huggingface.co/settings/tokens). Your
Hugging Face account must have accepted the
[Llama 3 license](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct),
or you can set `LLM_REPO_ID` to another text-generation model.

```bash
git clone https://github.com/belviskhoremk/Talk-to-your-pdf.git
cd Talk-to-your-pdf

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Optional: CPU-only PyTorch is much smaller to download
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env               # then add your token and a secret key
flask run
```

Open <http://127.0.0.1:5000>, upload a PDF and start asking questions.

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGINGFACEHUB_API_TOKEN` | ✅ | – | Hugging Face access token |
| `FLASK_SECRET_KEY` | ✅ | dev value | Signs the session cookie |
| `LLM_REPO_ID` | | `meta-llama/Meta-Llama-3-8B-Instruct` | Any HF text-generation model |
| `EMBEDDING_MODEL` | | `all-MiniLM-L6-v2` | Sentence-Transformers model |

## API

`POST /ask` (after uploading a PDF in the same browser session)

```json
// request
{ "question": "What is the main conclusion of the paper?" }
// response
{ "answer": "..." }
```

## Limitations

- Sessions and vector stores are kept **in memory**: they are lost on restart
  and are not shared between multiple server processes.
- Scanned PDFs (images only) contain no extractable text; they would need OCR first.
