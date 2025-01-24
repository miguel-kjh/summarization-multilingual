from collections import defaultdict
from typing import Dict
from datasets import Dataset
from tqdm import tqdm

from utils import INSTRUCTION_TEMPLATE, generate_training_prompt
from baseline import ExtractiveSummarizer



class TransformData:
    def __init__(self):
        self.template_json = {
            "instruction": "",
            "input": "",
            "output": "",
            "text": "",
            "language": ""
        }
        
    def generate_instructions(self, dataset: Dataset, lang: str) -> Dataset:
        instructions = []
        for sample in tqdm(dataset, desc=f"Generating instructions for {lang}"):
            template = self.template_json.copy()
            template['instruction'] = INSTRUCTION_TEMPLATE[lang]
            template['input'] = sample['text']
            template['output'] = sample['summary']
            template['text'] = generate_training_prompt(template['instruction'], template['input'], template['output'])
            template['language'] = lang
            instructions.append(template)

        print(f"Generated {len(instructions)} instructions for {lang}")
        return Dataset.from_list(instructions)
    
class TransformDataReduce(TransformData):
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        super().__init__()
        self.summarizer = ExtractiveSummarizer(model_name)

    def generate_instructions(self, dataset: Dataset, lang: str) -> Dataset:
        instructions = []
        for sample in tqdm(dataset, desc=f"Generating reduce instructions for {lang}"):
            template = self.template_json.copy()
            try:
                template['input'] = self.summarizer.summarize(sample['text'], lang)
            except Exception as e:
                print(f"Error summarizing {e}")
                continue
            template['instruction'] = INSTRUCTION_TEMPLATE[lang]
            template['output'] = sample['summary']
            template['text'] = generate_training_prompt(template['instruction'], template['input'], template['output'])
            template['language'] = lang
            instructions.append(template)
        print(f"Generated {len(instructions)} instructions for {lang}")
        return Dataset.from_list(instructions)
