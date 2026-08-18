# from dotenv import load_dotenv
# load_dotenv()

from langchain.chat_models import init_chat_model
 


from dotenv import load_dotenv

load_dotenv()

# from langchain_groq import ChatGroq

model = init_chat_model(
    "openai/gpt-oss-20b" ,
    model_provider="groq"
)

response = model.invoke("What is cricket?")

print(response.content)
# print(response)