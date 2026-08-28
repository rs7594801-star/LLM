from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/"

loader = WebBaseLoader(url)

docs = loader.load()

print("Number of documents:", len(docs))
print(docs[0].page_content)