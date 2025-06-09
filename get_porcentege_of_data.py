import argparse
import os
from transformers import AutoTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np

models = {
    "llama": "meta-llama/Llama-3.2-1B",
    "qwen":  "Qwen/Qwen3-0.6B",
}

def get_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path)

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))

if __name__ == "__main__":
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Get percentage of data from dataset")
        parser.add_argument("--dataset_path", type=str, default="data/02-processed/canario", help="Path to the dataset")
        parser.add_argument("--model", type=str, default="qwen", help="Model to use (llama or qwen)")
        return parser.parse_args()

    args = parse_arguments()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist.")
    
    print(f"Dataset path: {args.dataset_path}")
    tokenizer = get_tokenizer(models[args.model])
    data = load_from_disk(args.dataset_path)

    def format_text(sample):
        return sample['input'] + sample['output']
    
    data = data.map(lambda x: {"text": format_text(x)})

    list_of_tokens = [count_tokens(tokenizer, doc) for doc in data["train"]["text"]]
    total_tokens = sum(list_of_tokens)
    num_documents = len(list_of_tokens)
    average_tokens = total_tokens / num_documents if num_documents > 0 else 0

    print(f"Total documents: {num_documents}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per document: {average_tokens:.2f}")

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




    
