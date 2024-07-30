from Augmentation_processing import *
import torch
from Retrieval_processing import *
from sentence_transformers import SentenceTransformer,util
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Enter file path: ")
HF_token="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"
path=input()

def file_processing():

    embeddings,pages_and_chunks=retrieval_setup(pdf_file_path=path)
    print("--------------------")
    a=model_config()
    llm_model,tokenizer=a[0],a[1]
    data={"embeddings":embeddings,"pages_and_chunks":pages_and_chunks,"llm_model":llm_model,"tokenizer":tokenizer}
    
    return data
data=file_processing()

print("Enter query: ")
query=input()
output_text, context_items=ask_query(query=query,
                               llm_model=data["llm_model"],
                               tokenizer=data["tokenizer"],
                               embeddings=data["embeddings"],
                               pages_and_chunks=data["pages_and_chunks"])
print(output_text)

    