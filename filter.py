import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

FOLDER = "test_summary_truncate.xlsx"
target_tokens = 16384 - 2048  # 16k - 2k for the summary

if __name__ == "__main__":
    # Path to the directory containing the data
    base_dir = "models/baseline/spanish/ollama/qwen2.5:7b"
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

    # read in hugingface dataset
    folder = os.path.join(base_dir, FOLDER)
    df = pd.read_excel(folder, sheet_name="Sheet1")
    print(df)

    """print(dataset["document"][0])

    def count_tokens_in_dataset(example):
        return {"num_tokens": len(tokenizer(example["expected_summary"], add_special_tokens=False)["input_ids"])}
    dataset = dataset.map(count_tokens_in_dataset)
    dataset = dataset.filter(lambda x: x["num_tokens"] <= target_tokens)
    print(dataset["num_tokens"])"""

    from datasets import load_from_disk
    dataset = load_from_disk("data/02-processed/spanish")

    for expected_summary in df["expected_summary"]:
        print(f"Filtering for expected summary: {expected_summary}")
        resultados = dataset["test"].filter(lambda x: x["output"] == expected_summary)
    print(resultados["input"])
    

