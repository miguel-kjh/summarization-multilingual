from datasets import Dataset
from tqdm import tqdm

from utils import INSTRUCTION_TEMPLATE, SYSTEM_PROMPT



class TransformData:
    def __init__(self):
        self.template_json = {
            "system_prompt": "",
            "instruction": "",
            "input": "",
            "output": "",
            "language": ""
        }
        
    def generate_instructions(self, dataset: Dataset, lang: str) -> Dataset:
        instructions = []
        for sample in tqdm(dataset, desc=f"Generating instructions for {lang}"):
            template = self.template_json.copy()
            template['system_prompt'] = SYSTEM_PROMPT[lang]
            template['instruction'] = INSTRUCTION_TEMPLATE[lang]
            template['input'] = sample['text']
            template['output'] = sample['summary']
            template['language'] = lang
            instructions.append(template)

        print(f"Generated {len(instructions)} instructions for {lang}")
        return Dataset.from_list(instructions)
    
class TransformDataCanario(TransformData):

    def generate_instructions(self, dataset: Dataset) -> Dataset:
        instructions = []
        for sample in tqdm(dataset, desc=f"Generating instructions for canario"):
            template = self.template_json.copy()
            template['system_prompt'] = SYSTEM_PROMPT["canario"]
            template['instruction'] = INSTRUCTION_TEMPLATE["canario"]
            template['input'] = sample['original_text']
            template['output'] = sample['summarized_text']
            template['language'] = "canario"
            instructions.append(template)

        print(f"Generated {len(instructions)} instructions for canario")
        return Dataset.from_list(instructions)
