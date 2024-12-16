from typing import Tuple
import torch
from datasets import Dataset
from tqdm import tqdm
import time

from utils import SEED, generate_prompt

class SummaryGenerator:
    def __init__(self, tokenizer, device="cpu"):
        self.device    = device
        self.tokenizer = tokenizer

    def summarize(self, model, text: str, max_new_tokens: int = 256, temperature: float = 0.0001) -> Tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
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
            instruction, input, output, language = obj['instruction'], obj['input'], obj['text'], obj['language']
            try:
                prompt  = generate_prompt(instruction, input)
                summary, time = self.summarize(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                summaries.append({
                    'text': input, 
                    'generated_summary': summary,
                    'output': output,
                    'language': language,
                    'time': time,
                })
            except torch.cuda.OutOfMemoryError:
                pass
                #print("Out of memory error")
            torch.cuda.empty_cache()
        return summaries
