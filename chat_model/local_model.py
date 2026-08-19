from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" , 
    task="text-generation" ,
    pipeline_kwargs = dict(
        max_new_tokens = 512,
        do_sample = False ,
        repetition_penalty = 1.1 ,

    )
)

chat_model = ChatHuggingFace(llm = llm)

chat_model.invoke("what is data science?")

result = chat_model.invoke("my name is rachit ")
result = chat_model.invoke("from today onwards your name is TAASHU ")

print(result.content)