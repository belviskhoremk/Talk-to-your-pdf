"""Retrieval-Augmented Generation (RAG) pipeline for chatting with a PDF.

Pipeline:

1. **Load**: ``PyPDFLoader`` reads the PDF, one document per page.
2. **Split**: ``RecursiveCharacterTextSplitter`` cuts pages into overlapping
   chunks small enough to embed and to fit in the LLM prompt.
3. **Embed & store**: every chunk is embedded with a Sentence-Transformers
   model and stored in an in-memory Chroma vector database.
4. **Retrieve & answer**: for each question, a ``ConversationalRetrievalChain``
   rewrites the question using the chat history, fetches the most similar
   chunks, and asks the LLM to answer using them as context.
"""

from functools import lru_cache

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

import config

# Prompt used to turn a follow-up question plus chat history into a
# standalone question that can be used for retrieval.
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the conversation below and a follow-up question, rephrase the
follow-up question to be a standalone question.

Chat History:
{chat_history}
Follow-up question: {question}
Standalone question:"""
)


class SentenceTransformersEmbeddings:
    """Minimal LangChain-compatible wrapper around a Sentence-Transformers model.

    LangChain vector stores only need ``embed_documents`` and ``embed_query``.
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""
        return self.model.encode(text, convert_to_tensor=False).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> SentenceTransformersEmbeddings:
    """Load the embedding model once and share it between sessions."""
    return SentenceTransformersEmbeddings(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm() -> HuggingFaceEndpoint:
    """Create the Hugging Face LLM client once (lazily, so the app can start
    and show a clear error if the token is missing)."""
    if not config.HUGGINGFACEHUB_API_TOKEN:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is not set. Copy .env.example to .env "
            "and add your Hugging Face token."
        )
    return HuggingFaceEndpoint(
        repo_id=config.LLM_REPO_ID,
        huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
        task="text-generation",
        max_new_tokens=512,
    )


def build_qa_chain(pdf_path: str) -> ConversationalRetrievalChain:
    """Index ``pdf_path`` and return a conversational QA chain over it."""
    pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    vectordb = Chroma.from_documents(documents=chunks, embedding=get_embeddings())

    return ConversationalRetrievalChain.from_llm(
        get_llm(),
        retriever=vectordb.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
    )


def trim_history(
    history: list[tuple[str, str]], max_words: int = config.MAX_HISTORY_WORDS
) -> list[tuple[str, str]]:
    """Drop the oldest (question, answer) turns until the history fits ``max_words``."""

    def word_count(turns: list[tuple[str, str]]) -> int:
        return sum(len(q.split()) + len(a.split()) for q, a in turns)

    history = list(history)
    while history and word_count(history) > max_words:
        history.pop(0)
    return history


def ask(chain: ConversationalRetrievalChain, question: str,
        history: list[tuple[str, str]]) -> str:
    """Ask ``question`` about the indexed PDF and return the answer text."""
    result = chain.invoke({"question": question, "chat_history": history})
    # Some models echo the prompt; keep only the text after the last marker.
    return result["answer"].split("Helpful Answer:")[-1].strip()
