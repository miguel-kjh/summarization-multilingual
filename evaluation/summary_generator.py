from typing import Tuple
import torch
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_cluster import DocumentClusterer
from utils import SEED, generate_prompt

class SummaryGenerator:
    def __init__(self, tokenizer, device="cpu"):
        self.device    = device
        self.tokenizer = tokenizer

    def summarize(self, model, text: str, max_new_tokens: int = 256, temperature: float = 0.7) -> Tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            start = time.time()
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,  # Penaliza repetir palabras
                no_repeat_ngram_size=4,  # Evita repetir grupos de 4 palabras
                do_sample=True,          # Activar muestreo si no lo tienes ya por defecto
            )
            end = time.time()
            text = self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
        return text, end - start   

    def generate_summaries(self, model, dataset: Dataset, num_samples: int=5, max_new_tokens: int=256, temperature: float=0.7) -> list:
        summaries = []
        # get a subset of the dataset
        shuffle_dataset = dataset.shuffle(seed=SEED).select(range(num_samples))
        for obj in tqdm(shuffle_dataset, desc="Generating summaries"):
            instruction, input, output, language = obj['instruction'], obj['input'], obj['output'], obj['language']
            try:
                prompt  = generate_prompt(instruction, input)
                summary, time = self.summarize(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
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
