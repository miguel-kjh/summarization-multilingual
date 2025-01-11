
import os
import pandas as pd
import torch
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evaluation.summary_generator import SummaryGenerator
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_cluster import DocumentClustererKMeans, DocumentClustererTopKSentences

model_name = "models/Llama-3.2-1B-spanish_sentences_clustering-e1-b2-lr0.0001-wd0.0-c1024-r8-a16-d0.05-quant-2025-01-10-17-39-44"
dataset = "data/02-processed/spanish"
data_sample = 50
max_new_tokens = 512
using_clustering = "clf" # none, "topk", "clf"
cluster_embedding_model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
spacy_model = "es_dep_news_trf"

top_k_sents = 1
clasification_model = "models/RandomForest_best_model.pkl"

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

    num_samples = data_sample * dataset["test"].num_rows // 100

    if using_clustering:
        embedding_model = SentenceTransformer(cluster_embedding_model)

        if using_clustering == "topk":
            clusterer = DocumentClustererTopKSentences(embedding_model, spacy_model, top_k_sents=top_k_sents)
        elif using_clustering == "clf":
            loaded_model = joblib.load(clasification_model)
            clusterer = DocumentClustererKMeans(embedding_model, spacy_model, loaded_model)
        else:
            raise ValueError(f"Invalid clustering method: {using_clustering}")
        
        print("#"*10, f"Using clustering with {using_clustering}", "#"*10)
        summaries = summary_generator.generate_summaries_from_cluster(
            model,
            clusterer,
            dataset["test"],
            num_samples=num_samples, 
            max_new_tokens=max_new_tokens, 
        )
    else:
        print("#"*10, "Normal summarization", "#"*10)
        summaries = summary_generator.generate_summaries(
            model, 
            dataset["test"], 
            num_samples=num_samples, 
            max_new_tokens=max_new_tokens
        )
    
    
    df_summary = pd.DataFrame(summaries)
    name_df = f"test_summary_{using_clustering if using_clustering else 'normal'}.xlsx"
    df_summary.to_excel(os.path.join(model_name, name_df), index=False)
    print("Summaries generated")
