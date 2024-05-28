from flask import Flask, Response, render_template, url_for, request, jsonify
import requests
import json
import time
import base64
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from dotenv import load_dotenv
from operator import itemgetter
from fpdf import FPDF

import os
from PIL import Image
import pytesseract

load_dotenv()

app = Flask(__name__)
CORS(app)
API_KEY = os.getenv("OPENAI_API_KEY")


MODEL = 'gpt-4o'
model = ChatOpenAI(api_key=API_KEY, model=MODEL)
embeddings = OpenAIEmbeddings()
parser = StrOutputParser()

def extract_text_from_images(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Use pytesseract to extract text
                    text = pytesseract.image_to_string(img)
                    text2 = text.encode('latin-1', 'replace').decode('latin-1')
                    print(f"Text from {filename}:")
                    with open('generated.txt', 'a') as file:
                        file.write(text2)
            except Exception as e:
                print(f"Failed to process image {filename}: {e}")

def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size = 12)
    with open('sample.txt', 'r') as file:
        for line in file:
            pdf.cell(200, 10, txt = line, ln = True, align = 'L')
    # Save the PDF with name .pdf
    pdf_output = 'generated.pdf'
    pdf.output(pdf_output)

def load_pdf_memory():
    # load the pdf
    loader = PyPDFLoader("sample.pdf")
    pages = loader.load_and_split()
    return pages


def perform_rag(pages):
    template = """

    Answer the questions based on the context below. The context is full of code snippets. If you cannot, reply with "I don't know"

    Context: {context}
    Question: {question}

    """

    prompt = PromptTemplate.from_template(template)

    vectorstore = DocArrayInMemorySearch.from_documents(
        pages,
        embedding= embeddings
    )

    retriever = vectorstore.as_retriever()
    # retriever.invoke()

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question") 
        }
        | prompt
        | model
        | parser
    )

    res = chain.invoke({"question": "can you find the code snippet about mesh and return that as a reply?"})

    return res

@app.route('/',methods = ['GET', 'POST'])
def index():

    if request.method == "POST":
        data = request.json
        print(data)
        return jsonify({'message': 'Form received!', 'result': data})

    # folder_path = 'screenshots/'
    # extract_text_from_images(folder_path)
    # generate_pdf()
    # pages = load_pdf_memory()
    # perform_rag(pages)

    return render_template('index.html')













if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8081)