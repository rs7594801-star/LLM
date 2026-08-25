from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.9
)

print("Chose yoour ai mode")
print("1. HACKATHON")
print("2. ACADEMIC") 
print("3. PLACEMENT INTERVIEW")
print("4. PROFESSIONAL INTERACTION")
choice = input("Enter your choice (1-4): ")
if choice == "1":
    mode = "hackathon"
elif choice == "2":
    mode = "academic    "
elif choice == "3":
    mode = "placement"
elif choice == "4":
    mode = "professional"
messages =  [
    SystemMessage(content = mode)
]

print("-------------Hey welcome-----------------")

while True :
    # print("-------------Hey welcome-----------------")
    prompt = input("you : ")
    messages.append(HumanMessage(content = prompt))
    if prompt.lower() == "exit":
        print("Exiting the chat...")
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content = response.content))

    print("Bot : " , response.content) 