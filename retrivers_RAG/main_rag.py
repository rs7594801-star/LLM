from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

# from langchain_mistralai import MistralAIEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory ="chroma_db_2",
    embedding_function = embedding_model

)


retriever = vectorstore.as_retriever(
    search_type ="mmr",
    search_kwargs ={ 
        "k" : 4 ,
        "fetch_k" : 10 ,
        "lambda_mult" : 0.5}
    
)

llm = ChatMistralAI(model="mistral-small-latest")
prompt = ChatPromptTemplate.from_messages(
    [
    ("system",
        """you are a helpful ai assistant ,
    Use only provided context to answere the question .,
    if the answere is not present in the context ,
    say : "i could not find the ans int the document " """) ,
    (
        "human" ,
        """Context:
        {context}
        
        Question :
        {question}
        
        """

      

    )
    ]
)

print("Rag system created ")

print("Press exit to exit")

while True:
    query = input("you :")
    if query.lower() == "exit" :
        print("Rag system Closed .") 
        break 
    docs = retriever.invoke(query)
    print("\n--- RETRIEVED DOCUMENTS ---")

    for doc in docs:
        print(doc.page_content)
        print("--------------------------")

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")
# print("You asked:",query)