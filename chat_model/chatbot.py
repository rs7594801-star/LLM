from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)
messages =  [

]

print("-------------Hey welcome-----------------")

while True :
    # print("-------------Hey welcome-----------------")
    prompt = input("you : ")
    messages.append(prompt)
    if prompt.lower() == "exit":
        print("Exiting the chat...")
        break
    response = model.invoke(messages)
    messages.append(response.content)

    print("Bot : " , response.content)