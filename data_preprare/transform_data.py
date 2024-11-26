from collections import defaultdict
from typing import Dict
from datasets import Dataset
from tqdm import tqdm

from utils import INSTRUCTION_TEMPLATE



class TransformData:
    def __init__(self):
        self.template_json = {
            "instruction": "",
            "input": "",
            "output": "",
        }
        
    def generate_instructions(self, dataset: Dataset, lang: str) -> Dict:
        instructions = []
        for sample in tqdm(dataset, desc=f"Generating instructions for {lang}"):
            template = self.template_json.copy()
            template['instruction'] = INSTRUCTION_TEMPLATE[lang]
            template['input'] = sample['text']
            template['output'] = sample['summary']
            instructions.append(template)
        return instructions