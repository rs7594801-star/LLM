""" load the pdf
split into chunks
 cretae embeddings 
 store into chroma """

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv 

load_dotenv()

data = PyPDFLoader("../Document_loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000 ,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)
print("print number of chunks:" , len(chunks))

embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    # document = chunks ,
    persist_directory="chroma_db_2",
    embedding_function=embedding_model
)  

print("PDF stored successfully in chroma ")