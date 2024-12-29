
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evaluation.summary_generator import SummaryGenerator
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model_name = "models/Llama-3.2-1B-spanish-e10-b4-lr0.0001-wd0.0-c512-r8-a16-d0.05-quant-2024-12-12-13-19-24"
dataset = "data/02-processed/spanish"
data_sample = 50
max_new_tokens = 512
using_clustering = True
cluster_embedding_model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
chunk_size = 1000
chunk_overlap = 100
num_clusters = 10

#main
if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary_generator = SummaryGenerator(
        tokenizer, 
        device=device,
    )

    print("Generating")

    num_samples = 2 #data_sample * dataset["test"].num_rows // 100

    if using_clustering:
        print("#"*10, "Using clustering", "#"*10)
        embedding_model = SentenceTransformer(model_name)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len, 
            is_separator_regex=False
        )
        summaries = summary_generator.generate_summaries_from_cluster(
            model,
            embedding_model,
            text_splitter,
            dataset["test"], 
            num_samples=num_samples, 
            max_new_tokens=max_new_tokens, 
            num_clusters=num_clusters
        )
    else:
        summaries = summary_generator.generate_summaries(
            model, 
            dataset["test"], 
            num_samples=num_samples, 
            max_new_tokens=max_new_tokens
        )
    
    
    df_summary = pd.DataFrame(summaries)
    df_summary.to_excel(os.path.join(model_name, "test_summary.xlsx"), index=False)
    print("Summaries generated")
