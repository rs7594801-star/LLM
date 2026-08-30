""" load the pdf
split into chunks
 cretae embeddings 
 store into chroma """

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv 

load_dotenv()

data  = PyPDFLoader("Chunking/random_text_tokenization_practice.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000 ,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)
print("print number of chunks:" , len(chunks))

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(

    documents=chunks,
    embedding= embedding_model,
    persist_directory="chroma_db_2"
)   

print("PDF stored successfully in chroma ")