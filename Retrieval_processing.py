import fitz,torch
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
import re
import numpy as np
import os
from sentence_transformers import SentenceTransformer,util

def text_formatter(text: str) -> str:
  cleaned_text = text.replace("\n", " ").strip()
  return cleaned_text
def open_and_read_pdf(pdf_path) :
    if not isinstance(pdf_path, str):
        pdf=pdf_path
        pdf_file_path = 'data_pdf.pdf'  # Save path for the uploaded PDF
        with open(pdf_file_path, 'wb') as f:
            f.write(pdf.read())
        doc=fitz.open(pdf_path)
    else: 
        doc=fitz.open(pdf_path)
    pages_and_texts=[]
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_texts.append({"page_number": page_number-41,
                            "page_char_count":len(text),
                            "page_word_count": len(text.split(" ")),
                            "page_sentence_count_raw": len(text.split(". ")),
                            "page_token_count": len(text)/4,
                            "text":text})
    return pages_and_texts

def split_sentences_into_chunks(pages_and_texts,num_sentence_chunk_size: int=10,min_token_length:int = 30):
    #split the pdf into sentences using spacy
    nlp=English()

    nlp.add_pipe("sentencizer")

    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)

        item["sentences"] = [str(sentence) for sentence in item["sentences"]]

        item["page_sentence_count_spacing"] = len(item["sentences"])

    #divide sentences into chunks

    def split_list(input_list,split_size: int=num_sentence_chunk_size) :
        return [input_list[i:i+split_size+1] for i in range (0,len(input_list),split_size)]

    for item in tqdm(pages_and_texts):
        item["sentence_chunks"]=split_list(input_list=item["sentences"],split_size=num_sentence_chunk_size)

        item["num_chunks"] = len(item["sentence_chunks"])
    
    pages_and_chunks = []

    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict={}
            chunk_dict["page_number"]=item["page_number"]

            joined_sentence_chunk="".join(sentence_chunk).replace("  "," ").strip()
            joined_sentence_chunk = re.sub(r"\.([A-Z])",r'. \1',joined_sentence_chunk  )

            chunk_dict["sentence_chunk"]=joined_sentence_chunk

            chunk_dict["chunk_char_count"]=len(joined_sentence_chunk)

            chunk_dict["chunk_char_count"] = len([word for word in joined_sentence_chunk.split(" ")])

            chunk_dict["chunk_token_count"]=len(joined_sentence_chunk)/4

            pages_and_chunks.append(chunk_dict)
    
    #filter sentences with less than a certain small number of tokens. They don't have much information to store. 
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    return pages_and_chunks_over_min_token_len

def tokenizing_sentences(pages_and_chunks,filename:str="data_pdf.csv",model: str="all-mpnet-base-v2",device="cuda"):
    print(model)
    #tokenize the chunked sentences and save them in csv file
    embedding_model = SentenceTransformer(model_name_or_path=model, device=device)
    for item in tqdm(pages_and_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
    embeddings_df_save_path = filename
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

def retrieve_tokenized_dataframe(file_name:str="data_pdf.csv",device="cuda"):
    #load the tokenized dataframes
    text_chunks_and_embedding_df = pd.read_csv(file_name)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
    return embeddings,pages_and_chunks

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model_name : str="all-mpnet-base-v2",
                                n_resources_to_return: int=4,
                                print_time: bool=False,device="cuda"):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """
    embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device)
    # Embed the query
    query_embedding =embedding_model.encode(query,
                                   convert_to_tensor=True)

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

    return scores, indices

def retrieval_setup(pdf_file_path:str,device='cuda'):
    embed_path=pdf_file_path[:-4]+".csv"
    
    pdf_info=open_and_read_pdf(pdf_file_path)
    pages_and_chunks=split_sentences_into_chunks(pdf_info)
    if not os.path.exists(embed_path):
        tokenizing_sentences(pages_and_chunks=pages_and_chunks,filename=embed_path,device=device)
    return retrieve_tokenized_dataframe(file_name=embed_path,device=device)
    
    



