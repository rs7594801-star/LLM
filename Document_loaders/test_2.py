from langchain_community.document_loaders import TextLoader 

data = TextLoader("Document_loaders/notes.txt")
# data = TextLoader("Document_loaders/note.txt")
docs = data.load()

print(docs[0].page_content)

