# Talk to Your PDF

This project utilizes Llama3, HuggingFaceHub, and Langchain to enable communication with your PDF files. Simply upload a PDF file and you can begin asking questions to the chatbot.

## How It Works

1. **PDF Processing:** Upon uploading a PDF, we use PyPDFLoaded() to load it. The pages are then split using RecursiveCharacterTextSplitter(), with the splits stored in a Chroma vector database via a custom embedding class.
   
2. **Conversation Handling:** We create a ConversationalRetrievalChain() to retrieve necessary information from the vector database. This information is then passed to the language model (LLM) to generate a suitable response.

## Installation (Local)

To run this application locally, follow these steps:

- Clone the repository.
- Create a virtual environment:
  - `python3 -m venv venv`
  - `source venv/bin/activate`
- Install the project dependencies:
  - `pip install -r requirements.txt`
- Obtain and Configure Hugging Face Token:
  - Generate a free token at [Hugging Face Tokens](https://huggingface.co/settings/tokens).
  - Paste the token key into the code at line 21.
- Run the app:
  - `flask run`
  
## Note

Ensure you have a valid Hugging Face token before running this application.
