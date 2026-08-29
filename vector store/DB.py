from langchain_community.vectorstores import Chroma
# from langchain_openai  import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings


from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "python.txt"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "pandas.txt"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "deep_learning.txt"})
]

embeddings = MistralAIEmbeddings(model="mistral-embed")


vector_store = Chroma.from_documents(
    documents=  docs ,
    embedding= embeddings,
    persist_directory= "chroma-db"

)


results = vector_store.similarity_search(
    "What is Python used for?"
)

for result in results:
    print(result.page_content)
    print(result.metadata)
    
