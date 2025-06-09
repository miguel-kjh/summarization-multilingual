import re
from typing import Tuple
import torch
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import time
from transformers import TextStreamer
from utils import SEED, generate_prompt

def extract_clean_assistant_response(full_text: str) -> str:
    # Buscar el último bloque <|assistant|>
    assistant_start = full_text.rfind("assistant")
    if assistant_start == -1:
        assistant_content = full_text
    else:
        assistant_content = full_text[assistant_start + len("assistant"):]

    # Eliminar los bloques <think>...</think> si existen
    assistant_content = re.sub(r"<think>.*?</think>", "", assistant_content, flags=re.DOTALL)

    # Eliminar espacios extra al principio y al final
    return assistant_content.strip()

class SummaryGenerator:
    def __init__(self, tokenizer, device="cpu"):
        self.device    = device
        self.tokenizer = tokenizer

    def generate_summary_in_streamer(self, model, dataset: Dataset, sample_idx: int = 1,  max_new_tokens=100, temperature=0.7):
        """
        Generate summaries using a streamer for a specific sample in the dataset.
        """
        sample = dataset[sample_idx]
        prompt = sample['prompt']
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        streamer = TextStreamer(self.tokenizer)
        
        with torch.no_grad():
            start = time.time()
            for token in model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature = 0.7, top_p = 0.8, top_k = 20, # (normal)
                # temperature = 0.6, top_p = 0.95, top_k = 20, # (thinking)
                repetition_penalty = 1.0,
                streamer=streamer
            ):
                print(token)
            end = time.time()
        
        return end - start

    def summarize(self, model, text: str, max_new_tokens: int = 256, temperature: float = 0.7) -> Tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
                temperature=temperature,
                do_sample=True,          # Activar muestreo si no lo tienes ya por defecto
            )
            end = time.time()
            text = self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
        if self.tokenizer.chat_template:
            text = extract_clean_assistant_response(text)
        return text, end - start   

    def generate_summaries(self, model, dataset: Dataset, num_samples: int=5, max_new_tokens: int=256, temperature: float=0.7) -> list:
        summaries = []
        # get a subset of the dataset
        shuffle_dataset = dataset.shuffle(seed=SEED).select(range(num_samples))
        for obj in tqdm(shuffle_dataset, desc="Generating summaries"):
            prompt, input, output, language = obj['prompt'], obj['input'], obj['output'], obj['language']
            try:
                summary, time = self.summarize(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                print(summary)
                summaries.append({
                    'document': input, 
                    'expected_summary': output,
                    'generated_summary': summary,
                    'language': language,
                    'time': time,
                })
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
                #print("Out of memory error")
            torch.cuda.empty_cache()
        return summaries
    
    def generate_summaries_from_cluster(
            self, 
            model, 
            dataset: Dataset, 
            num_samples: int=5, 
            max_new_tokens: int=256, 
            temperature: float=0.7,
        ) -> list:

        summaries = []
        # get a subset of the dataset
        df = dataset.to_pandas()
        print(f"Number of samples: {num_samples}")
        df_subset = df[df["original_index_document"] <= num_samples]
        for index, group in df_subset.groupby('original_index_document'):
            sub_dataset = Dataset.from_pandas(group)
            join_summary = []
            times = []
            original_input = sub_dataset['original_document'][0]
            original_sum = sub_dataset['original_summary'][0]
            language = sub_dataset['language'][0]

            for obj in tqdm(sub_dataset, desc=f"Generating summaries for cluster {index}"):
                instruction, input = obj['instruction'], obj['input']
                prompt  = generate_prompt(instruction, input)
                try:
                    summary, time = self.summarize(
                        model, 
                        prompt, 
                        max_new_tokens=max_new_tokens, 
                        temperature=temperature
                    )
                    join_summary.append(summary)
                    times.append(time)
                except Exception as e:
                    print(e)
                    torch.cuda.empty_cache()
                    continue
                torch.cuda.empty_cache()

            summaries.append({
                'document': original_input, 
                'expected_summary': original_sum,
                'generated_summary': (" ".join(join_summary)).strip(),
                'language': language,
                'time': np.mean(times),
            })
        return summaries
