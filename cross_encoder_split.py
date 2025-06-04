
from datasets import load_from_disk, Dataset, DatasetDict
import argparse
import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

SEED = 42
text_splitter = SemanticChunker(OpenAIEmbeddings())
model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)

def align_chunks_map(example, threshold=0.7):
    document_text = example["input"]
    summary_text = example["output"]

    doc_chunks = [d.page_content for d in text_splitter.create_documents([document_text])]
    sum_chunks = [s.page_content for s in text_splitter.create_documents([summary_text])]
    pairs_all = [(s, d) for s in sum_chunks for d in doc_chunks]

    if not pairs_all:
        return {
            "instruction": [],
            "input": [],
            "output": [],
            "text": [],
            "language": [],
            "original_index_document": [],
            "original_document": [],
            "original_summary": [],
        }

    scores = model.predict(pairs_all)

    aligned_pairs = []
    for i, s_chunk in enumerate(sum_chunks):
        best_score = -1
        best_doc_idx = -1
        for j, d_chunk in enumerate(doc_chunks):
            idx_flat = i * len(doc_chunks) + j
            score = scores[idx_flat]
            if score > best_score:
                best_score = score
                best_doc_idx = j

        if best_score > threshold:
            aligned_pairs.append({
                "instruction": "Resume el siguiente fragmento legal de forma clara y concisa.",
                "input": doc_chunks[best_doc_idx],
                "output": s_chunk,
                "similarity": round(best_score, 3)
            })

    # Devolver columnas separadas para expandir fácilmente
    return {
        "instruction": [p["instruction"] for p in aligned_pairs],
        "input": [p["input"] for p in aligned_pairs],
        "output": [p["output"] for p in aligned_pairs],
        "similarity": [p["similarity"] for p in aligned_pairs],
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process constants for dataset and model configuration.")

    parser.add_argument("--dataset_path", type=str, default="data/02-processed/canario",
                        help="Path to the dataset to be processed.")
    parser.add_argument("--method", type=str, default="chunks",
                        help="Method to use for processing ('sentences', 'paragraphs'. 'chunks').")
    parser.add_argument("--percentage_of_data", type=float, default=0.01,
                        help="Percentage of data to process (e.g., 0.01 for 1%).")

    args = parser.parse_args()
    language = args.dataset_path.split("/")[-1]
    name_new_dataset = f"data/04-clustering/{language}-cross-encoder"
    args.name_new_dataset = name_new_dataset
    return args

def get_sample(dataset, num_samples):
    if num_samples:
        dataset = dataset.shuffle(seed=SEED).select(range(int(num_samples * dataset.num_rows)))
    return dataset


def main():
    args = parse_arguments()
    dataset = load_from_disk(args.dataset_path)

    train_dataset = get_sample(dataset["train"], args.percentage_of_data)
    print(f"Number of samples in train dataset: {train_dataset.num_rows}")
    new_train_dataset = train_dataset.map(
        align_chunks_map,
        remove_columns=train_dataset.column_names,
        desc="Aligning chunks in train dataset",
    )
    print(f"Number of samples in new train dataset: {new_train_dataset.num_rows}")
    validation_dataset = get_sample(dataset["validation"], args.percentage_of_data)
    print(f"Number of samples in validation dataset: {validation_dataset.num_rows}")
    new_validation_dataset = validation_dataset.map(
        align_chunks_map,
        remove_columns=validation_dataset.column_names,
        desc="Aligning chunks in validation dataset"
    )
    print(f"Number of samples in new validation dataset: {new_validation_dataset.num_rows}")
    test_dataset = get_sample(dataset["test"], args.percentage_of_data)
    print(f"Number of samples in test dataset: {test_dataset.num_rows}")
    new_test_dataset = test_dataset.map(
        align_chunks_map,
        remove_columns=test_dataset.column_names,
        desc="Aligning chunks in test dataset"
    )
    print(f"Number of samples in new test dataset: {new_test_dataset.num_rows}")

    new_dataset = DatasetDict({
        "train": new_train_dataset,
        "validation": new_validation_dataset,
        "test": new_test_dataset
    }) 

    new_dataset.save_to_disk(args.name_new_dataset)
    

if __name__ == '__main__':
    main()