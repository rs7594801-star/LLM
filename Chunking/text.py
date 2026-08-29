from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("Chunking/chunk.txt")
docs = data.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     separators = "" ,
#     chunk_size=10,
#     chunk_overlap=1
# )

splitter = CharacterTextSplitter(
    separator="" ,
    chunk_size=50,
    chunk_overlap=1
)

# chunks = text_splitter.split_documents(docs)
chunks = splitter.split_documents(docs)

print("Number of chunks:", len(chunks))

# for i, chunk in enumerate(chunks): 
#     print(f"\n--- Chunk {i} ---")
#     print(chunk.page_content)
#     print(len(chunks))
#     print(chunks)
for i in chunks:
    print(i.page_content)
    print()
    print()
    print()