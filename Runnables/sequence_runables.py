from dotenv import load_dotenv
load_dotenv()
#mistral api k liye 
from langchain_mistralai import ChatMistralAI 
# ye prmpt k liye hai 
from langchain_core.prompts import ChatPromptTemplate 

#this is for short outputs 
from langchain_core.output_parsers import StrOutputParser

#prompt tempelate
prompt = ChatPromptTemplate.from_template(
    "explain {topic} in simple words"
)

#model

model = ChatMistralAI(model = "mistral-small-2506")

#output parser
parser = StrOutputParser()

#step by step manual flow 
for_prompt = prompt.format_messages(topic ="machine learning")

#call the model manually 
response = model.invoke(for_prompt)

#parse the output manually 
final_output = parser.parse(response.content)

print(final_output)


chain = prompt| model | parser

result = chain.invoke("machine learning ")
print(result)