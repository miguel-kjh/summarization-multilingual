
import argparse
import os
import pandas as pd
from unsloth import FastLanguageModel
import torch
from evaluation.summary_generator import SummaryGenerator
from datasets import load_from_disk


from distutils.util import strtobool

from utils import CONTEXT_WINDOWS, seed_everything, SEED


def parse():
    parser = argparse.ArgumentParser(description="Script to generate summaries")

    parser.add_argument("--model_name_or_path", type=str, 
    default="models/Qwen/Qwen2.5-1.5B-Instruct/german/lora/Qwen2.5-1.5B-Instruct-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-18-19-11", help="Model name")
    parser.add_argument("--dataset", type=str, default="data/02-processed/german", help="Dataset path")
    parser.add_argument("--context_window", type=int, default=8192, help="Context window size")
    parser.add_argument("--using_streamer", type=lambda x: bool(strtobool(x)), default=False, help="Use streamer for generation")
    parser.add_argument("--using_clustering", type=lambda x: bool(strtobool(x)), default=False, help="Clustering method to use")
    parser.add_argument("--rewrite", type=lambda x: bool(strtobool(x)), default=False, help="Rewrite the summaries")

    parser.add_argument("--data_sample", type=int, default=10, help="Size of the data sample")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens")
    parser.add_argument("--quantization", type=lambda x: bool(strtobool(x)), default=False, help="Quantization")

    parser.add_argument("--cluster_embedding_model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="Embedding model for clustering")
    parser.add_argument("--spacy_model", type=str, default="es_dep_news_trf", help="SpaCy model")
    parser.add_argument("--top_k_sents", type=int, default=1, help="Number of top sentences to consider")
    parser.add_argument("--clasification_model", type=str, default="models/RandomForest_best_model.pkl", help="Path to the classification model")


    return parser.parse_args()

def create_model_and_tokenizer(args):

    context_window = next(
        (value for key, value in CONTEXT_WINDOWS.items() if key in args.model_name_or_path),
        None
    )

    # Lanzar excepción si no se encuentra una coincidencia
    if context_window is None:
        raise ValueError(f"Context window not found for model '{args.model_name_or_path}'. Please specify a valid model name.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name_or_path,
        max_seq_length = context_window,  # Context window size
        dtype = None,
        load_in_4bit = args.quantization, # quantization QLoRA 4-bit
    )
    return tokenizer, model

#main
if __name__ == '__main__':
    seed_everything(SEED)
    args = parse()
    name_df = f"test_summary_{'clustering' if args.using_clustering else 'normal'}.xlsx"

    if not os.path.exists(args.model_name_or_path):
        general_folder = "models/others"
        lang_folder    = args.dataset.replace("/", "_")
        final_folder   = os.path.join(general_folder, lang_folder, args.model_name_or_path)
        print(f"Creating folder {final_folder} for summaries")
        os.makedirs(final_folder, exist_ok=True)
        name_df_of_summaries = os.path.join(final_folder, name_df)
        print(f"Saving summaries to {final_folder}")
    else:
       name_df_of_summaries = os.path.join(args.model_name_or_path, name_df)
    
    if not args.rewrite and os.path.exists(name_df_of_summaries):
        print("Summaries already generated")
        exit()

    tokenizer, model = create_model_and_tokenizer(args)
    FastLanguageModel.for_inference(model)

    dataset = load_from_disk(args.dataset)

    ##########
    # Create prompts
    ##########

    if tokenizer.chat_template:
        print("Using chat template for inference formatting")
        
        def formatting_func_inference(example):
            instruction = example["instruction"]
            empty_prompt = f"{instruction}\n{{document}}\n"
            messages = [
                {"role": "system", "content": example["system_prompt"]},
                {"role": "user", "content": empty_prompt.format(document=example["input"])},
            ]
            return tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt = True, # Must add for generation
                enable_thinking = False, # Disable thinking
            )
        
        dataset["test"] = dataset["test"].map(lambda x: {"prompt": formatting_func_inference(x)})
    else:
        def formatting_prompts_inference(example):
            instruction = example["instruction"]
            empty_prompt = f"### Instruction:{instruction}\n### Input:{{document}}\n### Response:\n"
            prompts = []
            inference_prompt = empty_prompt.format(document=example["input"]).replace("\n", " ").strip()
            prompts.append(inference_prompt)
            return prompts
        
        dataset["test"] = dataset["test"].map(lambda x: {"prompt": formatting_prompts_inference(x)})

    def count_tokens_in_dataset(example):
        return {"num_tokens": len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])}
    dataset["test"] = dataset["test"].map(count_tokens_in_dataset)
    ##########

    device = "cuda" if torch.cuda.is_available() else "cpu"

    summary_generator = SummaryGenerator(
        tokenizer, 
        device=device,
    )

    print("Generating")

    if args.using_streamer:
        # get index of th sample whit minimum num_tokens
        min_idx = dataset["test"]["num_tokens"].index(min(dataset["test"]["num_tokens"]))
        print(f"Minimum number of tokens in dataset: {dataset['test']['num_tokens'][0]} at index {min_idx}")
        print("#"*10, "Streamer summarization", "#"*10)
        time = summary_generator.generate_summary_in_streamer(
            model, 
            dataset["test"],
            sample_idx=0,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Time taken for generation: {time} seconds")
    else:
        
        target_tokens = args.context_window - args.max_new_tokens
        # filter dataset to only include samples with num_tokens less than target_tokens
        dataset["test"] = dataset["test"].filter(lambda x: x["num_tokens"] <= target_tokens)
        print(f"Filtered dataset to {len(dataset['test'])} samples with num_tokens <= {target_tokens}")
        num_samples = len(dataset["test"])
        print("#"*10, "Normal summarization", "#"*10)
        summaries = summary_generator.generate_summaries(
            model, 
            dataset["test"], 
            num_samples=num_samples, 
            max_new_tokens=args.max_new_tokens,
            temperature=0.7
        )
    
        df_summary = pd.DataFrame(summaries)
        df_summary.to_excel(name_df_of_summaries, index=False)
        print(os.path.dirname(name_df_of_summaries))
