from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("Chunking/random_text_tokenization_practice.pdf")
docs = data.load()
splitter = TokenTextSplitter(
    chunk_size = 500 ,
    chunk_overlap = 10
)
chunks = splitter.split_documents(docs)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content)
print(len(docs))
print("number of chunks " , len(chunks))