
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

    parser.add_argument("--model_name_or_path", type=str, default="BSC-LT/salamandra-2b-instruct", help="Model name")
    parser.add_argument("--is_adapter", type=lambda x: bool(strtobool(x)), default=False, help="Is adapter model")
    parser.add_argument("--dataset", type=str, default="data/02-processed/canario", help="Dataset path")
    parser.add_argument("--context_window", type=int, default=8192, help="Context window size")
    parser.add_argument("--using_streamer", type=lambda x: bool(strtobool(x)), default=False, help="Use streamer for generation")
    parser.add_argument("--truncate", type=lambda x: bool(strtobool(x)), default=False, help="Truncate the input to fit the context window")
    parser.add_argument("--rewrite", type=lambda x: bool(strtobool(x)), default=False, help="Rewrite the summaries")

    parser.add_argument("--data_sample", type=int, default=10, help="Size of the data sample")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens")
    parser.add_argument("--quantization", type=lambda x: bool(strtobool(x)), default=False, help="Quantization")


    return parser.parse_args()

def truncate_text(text, tokenizer, max_tokens):
    # Tokeniza con truncamiento
    tokenized = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        return_tensors=None,  # para que devuelva dict con input_ids
        return_attention_mask=False,
        return_token_type_ids=False
    )
    # Decodifica para volver a obtener el texto
    truncated_text = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
    return truncated_text

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
        fast_inference= True,  # Enable fast inference
        max_seq_length = args.context_window,  # Context window size
        dtype = None,
        gpu_memory_utilization=0.5,  # GPU memory utilization
        load_in_4bit = args.quantization, # quantization QLoRA 4-bit
        float8_kv_cache=False,  # Enable float8 kv cache for faster inference
    )
    return tokenizer, model

#main
if __name__ == '__main__':
    seed_everything(SEED)
    args = parse()
    name_df = f"test_summary_{'truncate' if args.truncate else 'normal'}.xlsx"
    target_tokens = args.context_window - args.max_new_tokens

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

    if not tokenizer.chat_template:
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3",
            mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        )

    print("Using chat template for inference formatting")
    
    def formatting_func_inference(example):
        instruction = example["instruction"]
        empty_prompt = f"{instruction}\n{{document}}\n"
        if args.truncate:
            input_text = truncate_text(example["input"], tokenizer, target_tokens)
        else:
            input_text = example["input"]
        messages = [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": empty_prompt.format(document=input_text)},
        ]
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt = True, # Must add for generation
            enable_thinking = False, # Disable thinking
        )
    
    dataset["test"] = dataset["test"].map(lambda x: {"prompt": formatting_func_inference(x)})
        

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
            sample_idx=min_idx,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Time taken for generation: {time} seconds")
    else:
        # filter dataset to only include samples with num_tokens less than target_tokens
        if not args.truncate:
            print(f"Filtering dataset to samples with num_tokens <= {target_tokens}")
            dataset["test"] = dataset["test"].filter(lambda x: x["num_tokens"] <= target_tokens)
            print(f"Filtered dataset to {len(dataset['test'])} samples with num_tokens <= {target_tokens}")
        else:
            print(f"Using all samples in dataset but truncating them to {target_tokens} tokens")
        
        num_samples = len(dataset["test"])
        print("#"*10, "Normal summarization", "#"*10)
        summaries = summary_generator.generate_summaries(
            model, 
            dataset["test"], 
            num_samples=num_samples, 
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            adapter_name=args.model_name_or_path if args.is_adapter else "",
        )
    
        df_summary = pd.DataFrame(summaries)
        df_summary.to_excel(name_df_of_summaries, index=False)
        print(os.path.dirname(name_df_of_summaries))
