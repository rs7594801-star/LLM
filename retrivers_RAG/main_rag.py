from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
data = TextLoader("Document_loaders/notes.txt")
docs =data.load()
template = ChatPromptTemplate.from_messages([
    ("system","you are a AI that summarizea the text"),
    ("human","{data}")
])

model = ChatMistralAI(
    model = "mistral-small-2506",
    temperature = 0.9
)
prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)
print(result.content)