
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evaluation.summary_generator import SummaryGenerator
from datasets import load_from_disk

model_name = "models/Llama-3.2-3B-spanish-e10-b2-lr0.0001-wd0.0-c512-r8-a16-d0.05-quant-2024-12-14-20-10-15"
dataset = "data/02-processed/spanish"
data_sample = 50
max_new_tokens = 512

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

    summaries = summary_generator.generate_summaries(model, dataset["test"], num_samples=2, max_new_tokens=max_new_tokens)
    df_summary = pd.DataFrame(summaries)
    df_summary.to_excel(os.path.join(model_name, "test_summary.xlsx"), index=False)
    print("Summaries generated")
