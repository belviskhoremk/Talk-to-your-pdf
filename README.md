Talk to your pdf

In this project, I used Llama3, HuggingFaceHub and Langchain to make you talk with you pdf.
All wht you have to do is to upload a pdf file and you can start asking question to the chatbot.

Working:
	When the pdf is uploaded, we load it using PyPDFLoaded(). After that we split the pages using RecursiveCharacterTextSplitter() . The splits are then stored in a Chroma vector database usinng a custom embedding class.
	We then create the ConversationalRetrievalChain() that is used to retrieve the necessary information in the vector database and pass it to the llm to generate a proper text.

Install it locally:
    • Clone the repo
    • Create a virtual environment
        ◦ python3 -m venv venv
        ◦ source venv/bin/activate
    • Install the project dependencies
        ◦ pip install -r requirements.txt
    • Run the application
        ◦ flask run


Note: You need a Hugging Face token before running this application. 
If you do not have one, you can generate one for free on ‘https://huggingface.co/settings/tokens’, then paste the key in the code on line 21