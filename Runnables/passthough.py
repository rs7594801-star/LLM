from dotenv import load_dotenv 
load_dotenv()
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableParallel , RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

model = model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_retries=5
)
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system" , "you are a code generator"),
    ("human","{topic}") 

])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system","you are the helpful assistant who explains code in simpler way "),
    ("human","Explain the code in a simpler way:\n\n{code} ")

])

seq = code_prompt | model | parser | explain_prompt | model | parser

result = seq.invoke({"topic" : "wrote a code for palindrome in python "})

print("This is the whole resut portion",result)