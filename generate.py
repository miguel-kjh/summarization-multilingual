
import argparse
import os
import pandas as pd
import torch
from evaluation.summary_generator import SummaryGenerator
from datasets import load_from_disk


from distutils.util import strtobool

from utils import create_model_and_tokenizer, seed_everything, SEED


def parse():
    parser = argparse.ArgumentParser(description="Script to generate summaries")

    parser.add_argument("--model_name_or_path", type=str, default="models/others/Qwen2.5-0.5B-spanish-e2-b1-lr0.0001-wd0.0-c1024-r8-a16-d0.05-quant-2025-01-25-07-05-38", help="Model name")
    parser.add_argument("--dataset", type=str, default="data/04-clustering/spanish-paragraphs-sentence-transformers", help="Dataset path")
    parser.add_argument("--using_clustering", type=lambda x: bool(strtobool(x)), default=True, help="Clustering method to use")
    parser.add_argument("--rewrite", type=lambda x: bool(strtobool(x)), default=False, help="Rewrite the summaries")

    parser.add_argument("--data_sample", type=int, default=10, help="Size of the data sample")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens")
    parser.add_argument("--quantization", type=lambda x: bool(strtobool(x)), default=False, help="Quantization")

    parser.add_argument("--cluster_embedding_model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="Embedding model for clustering")
    parser.add_argument("--spacy_model", type=str, default="es_dep_news_trf", help="SpaCy model")
    parser.add_argument("--top_k_sents", type=int, default=1, help="Number of top sentences to consider")
    parser.add_argument("--clasification_model", type=str, default="models/RandomForest_best_model.pkl", help="Path to the classification model")


    return parser.parse_args()

#main
if __name__ == '__main__':
    seed_everything(SEED)
    args = parse()
    name_df = f"test_summary_{'clustering' if args.using_clustering else 'normal'}.xlsx"
    name_df_of_summaries = os.path.join(args.model_name_or_path, name_df)
    if not args.rewrite and os.path.exists(name_df_of_summaries):
        print("Summaries already generated")
        exit()

    tokenizer, model = create_model_and_tokenizer(args)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(args.dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary_generator = SummaryGenerator(
        tokenizer, 
        device=device,
    )

    print("Generating")


    if args.using_clustering:
        print("#"*10, f"Using clustering", "#"*10)
        idxs = max(set(dataset["test"]["original_index_document"]))
        num_samples = args.data_sample * idxs // 100
        summaries = summary_generator.generate_summaries_from_cluster(
            model,
            dataset["test"],
            num_samples=num_samples, 
            max_new_tokens=args.max_new_tokens, 
        )
    else:
        num_samples = args.data_sample * dataset["test"].num_rows // 100
        print("#"*10, "Normal summarization", "#"*10)
        summaries = summary_generator.generate_summaries(
            model, 
            dataset["test"], 
            num_samples=num_samples, 
            max_new_tokens=args.max_new_tokens,
        )
    
    
    df_summary = pd.DataFrame(summaries)
    df_summary.to_excel(name_df_of_summaries, index=False)
    print("Summaries generated")
