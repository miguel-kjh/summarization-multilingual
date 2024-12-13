import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import SEED, generate_prompt

class SummaryGenerator:
    def __init__(self, tokenizer, device="cpu"):
        self.device    = device
        self.tokenizer = tokenizer

    def summarize(self, model, text: str, max_new_tokens: int = 256, temperature: float = 0.0001) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
                temperature=temperature,
            )
        return self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
        #return self.tokenizer.batch_decode(outputs[0][inputs_length:], skip_special_tokens=True)

    def generate_summaries(self, model, dataset: Dataset, num_samples: int=5, max_new_tokens: int=256, temperature: float=0.0001) -> list:
        summaries = []
        # get a subset of the dataset
        shuffle_dataset = dataset.shuffle(seed=SEED)
        shuffle_dataset = shuffle_dataset[:num_samples]
        iterator_obj = zip(shuffle_dataset["instruction"], shuffle_dataset["input"], shuffle_dataset["output"], shuffle_dataset["language"])
        for obj in tqdm(iterator_obj, desc="Generating summaries"):
            instruction, input, output, language = obj
            try:
                prompt  = generate_prompt(instruction, input)
                summary = self.summarize(model, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                summaries.append({
                    'text': input, 
                    'generated_summary': summary,
                    'output': output,
                    'language': language,
                })
            except torch.cuda.OutOfMemoryError:
                print("Out of memory error")
            torch.cuda.empty_cache()
            print("Memory freed")
        return summaries
