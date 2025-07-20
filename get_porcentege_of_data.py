import argparse
import os
from transformers import AutoTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from nltk.util import ngrams
from nltk.tokenize import ToktokTokenizer
from tqdm import tqdm
from collections import Counter, defaultdict

models = {
    "llama": "meta-llama/Llama-3.2-1B",
    "qwen":  "Qwen/Qwen3-4B",
}

def get_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path)

def compute_ngram_set(tokens, n):
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()

def get_fragments(a_tokens, s_tokens):
    fragments = []
    i = 0
    while i < len(s_tokens):
        found = False
        for j in range(len(a_tokens)):
            k = 0
            while i + k < len(s_tokens) and j + k < len(a_tokens) and s_tokens[i + k] == a_tokens[j + k]:
                k += 1
            if k > 0:
                fragments.append(s_tokens[i:i+k])
                i += k
                found = True
                break
        if not found:
            i += 1
    return fragments

def compute_statistics_token_level(dataset, tokenizer):
    input_lengths = []
    summary_lengths = []
    summary_words_lengths = []
    new_ngrams_total = {1: [], 2: [], 3: [], 4: []}
    coverages = []
    densities = []
    compression_ratios = []
    toktok = ToktokTokenizer()

    for sample in tqdm(dataset, desc="Computing statistics"):
        input_text = sample["input"]
        summary_text = sample["output"]

        input_tokens = tokenizer.tokenize(input_text.lower())
        summary_tokens = tokenizer.tokenize(summary_text.lower())
        summary_words = toktok.tokenize(summary_text.lower())

        input_lengths.append(len(input_tokens))
        summary_lengths.append(len(summary_tokens))
        summary_words_lengths.append(len(summary_words))
        compression_ratios.append(len(input_tokens) / len(summary_tokens) if summary_tokens else 0)

        for n in range(1, 5):
            summary_ngrams = compute_ngram_set(summary_tokens, n)
            input_ngrams = compute_ngram_set(input_tokens, n)
            new_ngrams = summary_ngrams - input_ngrams
            pct_new = len(new_ngrams) / len(summary_ngrams) * 100 if summary_ngrams else 0
            new_ngrams_total[n].append(pct_new)

        fragments = get_fragments(input_tokens, summary_tokens)
        frag_lens = [len(f) for f in fragments]

        coverage = sum(frag_lens) / len(summary_tokens) if summary_tokens else 0
        density = sum(l**2 for l in frag_lens) / len(summary_tokens) if summary_tokens else 0

        coverages.append(coverage)
        densities.append(density)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    results = {
        "Input tokens avg": avg(input_lengths),
        "Summary tokens avg": avg(summary_lengths),
        "Summary words avg": avg(summary_words_lengths),
        "Compression ratio": avg(compression_ratios),
        "Coverage": avg(coverages),
        "Density": avg(densities),
    }

    for n in range(1, 5):
        results[f"New {n}-grams (%)"] = avg(new_ngrams_total[n])

    return results

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Get percentage of new n-grams in dataset")
        parser.add_argument("--dataset_path", type=str, default="data/02-processed/english", help="Path to the dataset")
        parser.add_argument("--model", type=str, default="qwen", help="Model to use (llama or qwen)")
        return parser.parse_args()

    models = {
        "llama": "meta-llama/Llama-3.2-1B",
        "qwen":  "Qwen/Qwen3-4B",
    }

    args = parse_arguments()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist.")
    
    print(f"Dataset path: {args.dataset_path}")
    tokenizer = get_tokenizer(models[args.model])
    data = load_from_disk(args.dataset_path)
    # Verifica que existen los splits
    splits = []
    for split_name in ["train", "test", "validation"]:
        if split_name in data:
            splits.append(data[split_name])

    # Concatena los splits en un solo dataset
    if splits:
        from datasets import concatenate_datasets
        new_data = concatenate_datasets(splits)
        print(new_data)
    else:
        raise ValueError("No se encontraron splits válidos en el dataset.")

    stats = compute_statistics_token_level(new_data, tokenizer)

    for k, v in stats.items():
        print(f"{k}: {v:.2f}")
    

    # save the statistics to a text file
    output_file = os.path.join(args.dataset_path, "statistics.txt")
    with open(output_file, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.2f}\n")
    print(f"Statistics saved to {output_file}") 

    exit(0)
    # Mejorar el histograma
    plt.figure(figsize=(10, 6))
    plt.hist(list_of_tokens, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
    plt.axvline(average_tokens, color='red', linestyle='dashed', linewidth=2, label=f"Media = {average_tokens:.2f}")
    plt.axvline(8192, color='blue', linestyle='dashed', linewidth=2, label=f"Ventana de contexto de entrenamiento")
    plt.axvline(120000, color='green', linestyle='dashed', linewidth=2, label=f"Ventana de contexto de Llama")
    plt.axvline(40000, color='black', linestyle='dashed', linewidth=2, label=f"Ventana de contexto de Qwen")
    plt.title(f"Distribución de Tokens por Documento Parcan", fontsize=14)
    plt.xlabel("Número de Tokens", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Mostrar valores extremos
    min_tokens = min(list_of_tokens)
    max_tokens = max(list_of_tokens)
    plt.annotate(f"Min: {min_tokens}", xy=(min_tokens, 1), xytext=(min_tokens + 5, 5),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10)
    plt.annotate(f"Max: {max_tokens}", xy=(max_tokens, 1), xytext=(max_tokens - 100, 5),
                 arrowprops=dict(facecolor='orange', shrink=0.05), fontsize=10)

    print("Histogram displayed successfully.")
    # Save the histogram as an image
    histogram_path = os.path.join(args.dataset_path, "token_distribution_histogram.png")
    plt.savefig(histogram_path)
    plt.show()
    print(f"Histogram saved to {histogram_path}")
    plt.close()  # Close the plot to free memory




    
