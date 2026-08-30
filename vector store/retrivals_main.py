from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter 
load_dotenv()
data = PyPDFLoader("Chunking/tokenization_rag_practice_10_pages.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000 ,
    chunk_overlap = 200
)

# chunks = splitter.split_documents(docs)
docs =data.load()
chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages([
    ("system","you are a AI that summarizea the text"),
    ("human","{data}")
])

model = ChatMistralAI(
    model = "mistral-small-2506",
    temperature = 0.9
)
prompt = template.format_messages(data = docs[0].page_content)
prompt_chunks = template.format_messages(data = chunks[0].page_content)

result = model.invoke(prompt)
result = model.invoke(prompt_chunks)
print(result.content)
