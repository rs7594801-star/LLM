from dotenv import load_dotenv 
laod_dotenv()

from langchain.chat_models import ChatMistralAI

model = ChatMistralAI(model = "mistral-small-2506" , temperature = 0.9)

response = model.invoke("write a poem ")
print(response.content)