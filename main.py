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
API_KEY = os.getenv("OPENAI_API_KEY")


MODEL = 'gpt-4o'
# MODEL = 'llama3:8b'

if MODEL.startswith('gpt'):
    model = ChatOpenAI(api_key=API_KEY,model = MODEL)
    embeddings = OpenAIEmbeddings()
else:
    model = Ollama(model = MODEL)


parser = StrOutputParser()


# loader = PyPDFLoader('test.pdf')
# pages = loader.load()
# print(pages)

# print(chain.invoke("Tell me a joke"))


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
                    # print(text)
                    # print("\n" + "-"*50 + "\n")
                    with open('sample.txt', 'a') as file:
                        file.write(text2)
            except Exception as e:
                print(f"Failed to process image {filename}: {e}")

# Specify the path to your folder containing images
folder_path = 'screenshots/'
extract_text_from_images(folder_path)

pdf = FPDF()
pdf.add_page()
pdf.set_font("helvetica", size = 12)
with open('sample.txt', 'r') as file:
    for line in file:
        pdf.cell(200, 10, txt = line, ln = True, align = 'L')
# Save the PDF with name .pdf
pdf_output = 'sample.pdf'
pdf.output(pdf_output)



# load the pdf
loader = PyPDFLoader("sample.pdf")
pages = loader.load_and_split()

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

res = chain.invoke({"question": "can you find the code snippet about GroundMesh and return that as a reply?"})

print(res)