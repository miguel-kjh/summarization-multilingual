from collections import defaultdict
from typing import Dict
from datasets import Dataset
from tqdm import tqdm

from utils import INSTRUCTION_TEMPLATE, generate_training_prompt



class TransformData:
    def __init__(self):
        self.template_json = {
            "instruction": "",
            "input": "",
            "output": "",
            "text": "",
        }
        
    def generate_instructions(self, dataset: Dataset, lang: str) -> Dataset:
        instructions = []
        for sample in tqdm(dataset, desc=f"Generating instructions for {lang}"):
            template = self.template_json.copy()
            template['instruction'] = INSTRUCTION_TEMPLATE[lang]
            template['input'] = sample['text']
            template['output'] = sample['summary']
            template['text'] = generate_training_prompt(template['instruction'], template['input'], template['output'])
            instructions.append(template)

        print(f"Generated {len(instructions)} instructions for {lang}")
        return Dataset.from_list(instructions)