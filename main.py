from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

API_KEY = open("API_KEY", 'r').read()

# MODEL = 'gpt-4o'
MODEL = 'llama3:8b'

if MODEL.startswith('gpt'):
    model = ChatOpenAI(api_key=API_KEY,model = MODEL)
else:
    model = Ollama(model = MODEL)


parser = StrOutputParser()
chain = model | parser

loader = PyPDFLoader('test.pdf')
pages = loader.load()
print(pages)

# print(chain.invoke("Tell me a joke"))


