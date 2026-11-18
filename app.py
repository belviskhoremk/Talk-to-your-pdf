"""Flask web app: upload a PDF, then chat with it.

Routes:
    GET  /         Upload page.
    POST /upload   Index the uploaded PDF and open a chat session.
    GET  /chatbot  Chat page (requires an indexed PDF).
    POST /ask      JSON ``{"question": "..."}`` -> ``{"answer": "..."}``.

Run:
    flask run        # or: python app.py
"""

import os
import uuid

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

import config
import rag

app = Flask(__name__)
app.config["SECRET_KEY"] = config.FLASK_SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# In-memory store: session id -> {"chain": ..., "history": [(q, a), ...]}.
# Fine for a local demo; use a real store (e.g. Redis) for multiple workers.
user_sessions: dict[str, dict] = {}


def _current_session() -> dict | None:
    """Return the chat state of the current browser session, if any."""
    return user_sessions.get(session.get("session_id", ""))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # secure_filename strips path components such as "../" from the name.
    # The uuid prefix keeps two uploads with the same name from colliding.
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    file_path = os.path.join(config.UPLOAD_FOLDER, filename)
    file.save(file_path)

    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    try:
        chain = rag.build_qa_chain(file_path)
    except RuntimeError as exc:  # e.g. missing Hugging Face token
        return jsonify({"error": str(exc)}), 500
    user_sessions[session_id] = {"chain": chain, "history": []}

    return redirect(url_for("chatbot"))


@app.route("/chatbot")
def chatbot():
    if _current_session() is None:
        return redirect(url_for("index"))
    return render_template("chatbot.html")


@app.route("/ask", methods=["POST"])
def ask_question():
    state = _current_session()
    if state is None:
        return jsonify({"error": "No PDF processed yet"}), 400

    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400

    # Keep the prompt within the model's context window.
    state["history"] = rag.trim_history(state["history"])
    answer = rag.ask(state["chain"], question, state["history"])
    state["history"].append((question, answer))

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
