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

    def summarize(self, model, text: str, max_new_tokens: int = 256, temperature: float = 0.0001) -> Tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            start = time.time()
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
                temperature=temperature,
            )
            end = time.time()
            text = self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
        return text, end - start   

    def generate_summaries(self, model, dataset: Dataset, num_samples: int=5, max_new_tokens: int=256, temperature: float=0.0001) -> list:
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
            embedding_model: SentenceTransformer,
            spacy_model: str,
            top_k_sents: int,
            dataset: Dataset, 
            num_samples: int=5, 
            max_new_tokens: int=256, 
            temperature: float=0.0001,
        ) -> list:

        document_clusterer = DocumentClusterer(
            embedding_model, 
            spacy_model, 
            top_k_sents=top_k_sents,
        )
        summaries = []
        # get a subset of the dataset
        shuffle_dataset = dataset.shuffle(seed=SEED).select(range(num_samples))
        for obj in tqdm(shuffle_dataset, desc="Generating summaries"):
            instruction, input, output, language = obj['instruction'], obj['input'], obj['output'], obj['language']
            result = document_clusterer.cluster_and_assign(input)
            #join_summary = []
            #times = []
            #for (doc_parts, _) in result:
            #    text = " ".join(doc_parts)
            try:
                prompt  = generate_prompt(instruction, result)
                summary, time = self.summarize(
                    model, 
                    prompt, 
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature
                )
                #join_summary.append(summary)
                #times.append(time)
            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()
            summaries.append({
                'document': input, 
                'expected_summary': output,
                'generated_summary': summary, #" ".join(join_summary),
                'language': language,
                'time': time, #np.mean(times),
            })
        return summaries
