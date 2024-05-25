from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

import os
from PIL import Image
import pytesseract


API_KEY = open("API_KEY", 'r').read()


# MODEL = 'gpt-4o'
MODEL = 'llama3:8b'

if MODEL.startswith('gpt'):
    model = ChatOpenAI(api_key=API_KEY,model = MODEL)
else:
    model = Ollama(model = MODEL)


# parser = StrOutputParser()
# chain = model | parser

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
                    print(f"Text from {filename}:")
                    print(text)
                    print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"Failed to process image {filename}: {e}")

# Specify the path to your folder containing images
folder_path = 'screenshots/'
extract_text_from_images(folder_path)