from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI
import gradio
import os

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    # set number of output tokens
    num_outputs = 256

    _llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=_llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
    
    #Directory in which the indexes will be stored
    index.storage_context.persist(persist_dir="indexes")

    return index

def chatbot(input_text):
    
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="indexes")
    
    #load indexes from directory using storage_context 
    query_engne = load_index_from_storage(storage_context).as_query_engine()
    
    response = query_engne.query(input_text)
    
    #returning the response
    return response.response

#Creating the web UIusing gradio
iface = gradio.Interface(fn=chatbot,
                     inputs=gradio.inputs.Textbox(lines=5, label="Enter your question here"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

#Constructing indexes based on the documents in traininData folder
#This can be skipped if you have already trained your app and need to re-run it
index = construct_index("trainingData")

#launching the web UI using gradio
iface.launch(share=True)
