from dotenv import load_dotenv
load_dotenv()
#mistral api k liye 
from langchain_mistralai import ChatMistralAI 
# ye prmpt k liye hai 
from langchain_core.prompts import ChatPromptTemplate 

#this is for short outputs 
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from langchain_core.runnables import RunnableParallel , RunnableLambda

#prompt tempelate
short_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in one or two lines "
)
detailed_prompt = ChatPromptTemplate.from_template(
    "explain this {topic} in detail "
)

# from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_retries=5
)
parser = StrOutputParser()

topic = "machine learning"

#parallel runables 
#we have to place this in list 


chain = RunnableParallel({
    "short" :RunnableLambda(lambda x : x['short'])  | short_prompt | model | parser
    ,
    "detailed" :RunnableLambda (lambda x : x['detailed']) |detailed_prompt | model | parser
}) 


result = chain.invoke({"short" : {"topic" : "machine learning "} 
                       ,
                       "detailed" : {"topic" :"deep learnig "}
                       })

print(result)

print("SHORT")
print(result['short'])
print("DETAILED")
print(result['detailed'])