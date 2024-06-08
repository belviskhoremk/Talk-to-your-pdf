import langchain.chains.summarize
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import uuid
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts.prompt import PromptTemplate
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your_secret_key'  # Needed for session management
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
hugging_face_api_key = ""


class SentenceTransformersEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, documents):
        embeddings = self.model.encode(documents, convert_to_tensor=False)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query):
        embedding = self.model.encode(query, convert_to_tensor=False)
        return embedding.tolist()


embedding = SentenceTransformersEmbeddings(embedding_model)

template = """
  
  Chat History:
  {chat_history}
  Question: 
  {question}
  Helpful Answer:"""
QA_PROMPT = PromptTemplate.from_template(template)

llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=hugging_face_api_key,
                     task='text-generation')


# In-memory storage for session data
user_sessions = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(pages)

        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

        qa_chain_conv = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(),
            condense_question_prompt=QA_PROMPT,
            return_source_documents=True,
        )

        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        # Store in user_sessions
        user_sessions[session_id] = {
            'vectordb': vectordb,
            'qa_chain_conv': qa_chain_conv,
            'chat_history': []
        }

        return redirect(url_for('chatbot'))


@app.route('/chatbot')
def chatbot():
    if 'session_id' not in session or session['session_id'] not in user_sessions:
        return redirect(url_for('index'))
    return render_template('chatbot.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    if 'session_id' not in session or session['session_id'] not in user_sessions:
        return jsonify({'error': 'No PDF processed yet'})

    session_id = session['session_id']
    data = request.get_json()
    question = data.get('question')
    user_data = user_sessions[session_id]

    qa_chain_conv = user_data.get('qa_chain_conv')
    chat_history = user_data.get('chat_history')

    # Truncate chat history if it gets too long
    max_chat_history_tokens = 1024
    chat_history_str = " ".join([f"Human: {q} Assistant: {a}" for q, a in chat_history])
    while len(chat_history_str.split()) > max_chat_history_tokens:
        chat_history.pop(0)
        chat_history_str = " ".join([f"Human: {q} Assistant: {a}" for q, a in chat_history])

    result = qa_chain_conv({"question": question, "chat_history": chat_history})
    answer = result['answer'].split('Helpful Answer:')[-1].strip()
    chat_history.append((question, answer))
    user_sessions[session_id]['chat_history'] = chat_history

    return jsonify({'answer': answer})



if __name__ == '__main__':
    app.run(debug=True)
