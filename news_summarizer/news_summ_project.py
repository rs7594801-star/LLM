from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


search_tool = TavilySearchResults(mac_result = 5)


llm   = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_retries=5
)

prompt = ChatPromptTemplate.from_template(

"""
You are a helpful assistant 

summarize the following new into clear bullets points 

{news}



"""
)

chain = prompt | llm | StrOutputParser()

newa_result = search_tool.run("Latest AI news of 2026")

result = chain.invoke({"news" : newa_result})

print(result)
