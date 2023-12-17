from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import logging
import sys
import torch
from llama_index.llms import LlamaCPP

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("content/Data/").load_data()


from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
llm = LlamaCPP(
    model_url=None,
    
    model_path='model/mistral-7b-instruct-v0.1.Q4_K_S.gguf',
    temperature=0.1,
    max_new_tokens=256,
    
    context_window=3900,
    
    generate_kwargs={},
    
    
    model_kwargs={"n_gpu_layers": -1},
    
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext, set_global_service_context

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("What is career counselling?")

print(response)