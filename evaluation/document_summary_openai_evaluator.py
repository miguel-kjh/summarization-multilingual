import json
import os
import re
from langchain_ollama import OllamaLLM
import numpy as np
from openai import OpenAI

class DocumentSummaryOpenAiEvaluator:
    def __init__(self, api_key, upgrade=False):
        """
        Initialize the evaluator.

        :param api_key: OpenAI API key.
        :param prompt_files: Dictionary mapping evaluation criteria to their respective prompt files. 
                            Example: {"coherence": "coh.txt", "consistency": "con.txt", "fluency": "flu.txt", "relevance": "rel.txt"}
        """
        self.client = OpenAI(
            api_key=api_key,  # This is the default and can be omitted
        )
        self.prompt_files = {
            "coherence": "coh.txt",
            "consistency": "con.txt",
            "fluency": "flu.txt",
            "relevance": "rel.txt"
        }
        self.upgrade = upgrade
        print("OpenAI evaluator initialized.")
        print(f"Upgrade: {upgrade}")

    def _load_prompts(self, language="spanish"):
        """
        Load prompts from text files.

        :param prompt_files: Dictionary mapping evaluation criteria to file paths.
        :return: Dictionary mapping evaluation criteria to their respective prompts.
        """
        prompts = {}
        for criteria, file_path in self.prompt_files.items():
            file_path = os.path.join("evaluation", "prompts", language, file_path)
            with open(file_path, 'r') as file:
                prompts[criteria] = file.read()
        return prompts

    def evaluate(self, document, summary, language="spanish"):
        """
        Evaluate a summary against a document for all criteria.

        :param document: The source document.
        :param summary: The summary to evaluate.
        :return: Dictionary with scores for each evaluation criterion.
        """
        results = {}
        prompts = self._load_prompts(language)
        for criteria, prompt in prompts.items():
            evaluation_prompt = prompt.replace("{{Document}}", document).replace("{{Summary}}", summary)
            response = self._call_api(evaluation_prompt)
            results[criteria] = self._parse_response(response)
        return results

    def _call_api(self, prompt):
        """
        Call the OpenAI API with the given prompt.

        :param prompt: The prompt to send to the API.
        :return: The API response.
        """
        plus = "Be very generous when scoring." if self.upgrade else ""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator for document summaries" + plus},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        return response

    def _parse_response(self, response):
        """
        Parse the API response to extract the score.

        :param response: The API response.
        :return: The extracted score.
        """
        content = response.choices[0].message.content
        matched = re.search("\w+:\s*\[([\d.]+)\]", content)
        if (matched):
            try:
                score = float(matched.group(1))
            except:
                score = -1
        else:
            score = -1
        return score
    
class DocumentSummaryOllamaEvaluator(DocumentSummaryOpenAiEvaluator):
    def __init__(self, api_key = None, upgrade=False):
        self.llm = OllamaLLM(model="phi4")
        self.prompt_files = {
            "coherence": "coh.txt",
            "consistency": "con.txt",
            "fluency": "flu.txt",
            "relevance": "rel.txt"
        }
        self.upgrade = upgrade

    def evaluate(self, document, summary, language="spanish"):
        """
        Evaluate a summary against a document for all criteria.

        :param document: The source document.
        :param summary: The summary to evaluate.
        :return: Dictionary with scores for each evaluation criterion.
        """
        results = {}
        prompts = self._load_prompts(language)
        for criteria, prompt in prompts.items():
            evaluation_prompt = prompt.replace("{{Document}}", document).replace("{{Summary}}", summary)
            response = self.llm.invoke(evaluation_prompt)
            results[criteria] = self._parse_response(response)
        return results

    

# Example usage
if __name__ == "__main__":
    prompt_files = {
        "coherence": "coh.txt",
        "consistency": "con.txt",
        "fluency": "flu.txt",
        "relevance": "rel.txt"
    }
    language = "spanish"

    api_key_file = "api/key.json"
    with open(api_key_file, "r") as file:
        api_key = json.load(file)["key"]

    evaluator = DocumentSummaryOpenAiEvaluator(api_key, upgrade=True)

    # Sample document and summary
    file_qwq_trained = "models/Qwen/Qwen2.5-3B/italian-chunks-sentence-transformers/lora/Qwen2.5-3B-italian-chunks-sentence-transformers-e2-b1-lr0.0001-wd0.0-c256-peft-lora-r8-a16-d0.05-2025-02-24-22-12-13/test_summary_clustering.xlsx"

    #file_phi4 = "models/baseline/spanish/ollama/phi4/test_summary_normal.xlsx"

    #file_openai = "models/baseline/spanish/openai/test_summary_normal.xlsx"
    import pandas as pd
    df = pd.read_excel(file_qwq_trained, engine='openpyxl')
    print(df.head())
    document = df["document"].to_list()
    summary = df["generated_summary"].to_list()
    mean_socore = {
        "coherence": [],
        "consistency": [],
        "fluency": [],
        "relevance": [],
        "average": [],
    }

    for doc, sum in zip(document, summary):
        results = evaluator.evaluate(doc, sum, language=language)
        print(results)

        def calculate_weighted_mean(metrics: dict) -> float:
            """
            Calculate a normalized mean for metrics with different ranges.

            :param metrics: Dictionary of metrics and their values.
            :param max_values: Dictionary of metrics and their maximum possible values.
            :return: scaled mean (1-max of scale).
            """
            max_values = {'coherence': 5.0, 'consistency': 5.0, 'fluency': 3.0, 'relevance': 5.0}
            normalized_values = [value / max_values[metric] for metric, value in metrics.items()]
            normalized_mean = np.mean(normalized_values)
            
            # Optionally scale the mean to a desired range (e.g., 1 to 5)
            scaled_mean = normalized_mean * max(max_values.values())

            return scaled_mean

        scaled_mean = calculate_weighted_mean(results)
        print(f"Scaled Mean (1-5): {scaled_mean}")

        if scaled_mean >= 0:
            for key, value in results.items():
                mean_socore[key].append(value)
            mean_socore["average"].append(scaled_mean)

    for key, value in mean_socore.items():
        print(f"{key}: {np.mean(value)}")
