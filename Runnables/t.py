import time
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template("explain {topic} in simple words")

model = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.7
)

parser = StrOutputParser()

# --- Step-by-step manual flow ---
for_prompt = prompt.format_messages(topic="machine learning")
response = model.invoke(for_prompt)
final_output = parser.parse(response.content)
print("Manual Output:", final_output)

# Pause to respect free tier rate limits
time.sleep(3)

# --- LCEL Chain flow ---
chain = prompt | model | parser

# Note: Passed as a dictionary matching {topic}
result = chain.invoke({"topic": "machine learning"})
print("\nChain Output:", result)