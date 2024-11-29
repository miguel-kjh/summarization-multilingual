import torch
from datasets import Dataset

from utils import generate_prompt

class SummaryGenerator:
    def __init__(self, model, tokenizer, system_prompt, device="cpu"):
        self.model         = model
        self.device        = device
        self.tokenizer     = tokenizer
        self.system_prompt = system_prompt

    def summarize(self, text: str, max_new_tokens: int = 256, temperature: float = 0.0001) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature
            )
        return self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    def generate_summaries(self, dataset: Dataset, num_samples: int=5):
        summaries = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            print(f"Processing sample {i}")
            prompt = generate_prompt(example['text'])
            summary = self.summarize(prompt)
            summaries.append({
                'text': example['text'], 
                'generated_summary': summary
            })
        return summaries
