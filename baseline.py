import os
import time
from datasets import load_from_disk
import pandas as pd
from sentence_transformers import SentenceTransformer
from summarizer import Summarizer
from tqdm import tqdm
from transformers import *
from openai import OpenAI
from datasets import Dataset
from langchain_ollama import OllamaLLM


from document_cluster import DocumentClustererTopKSentences
from text2embeddings import Text2EmbeddingsSetenceTransforms

def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--dataset', type=str, default="data/02-processed/spanish")
    parser.add_argument('--method', type=str, default="ollama")
    parser.add_argument('--model_name', type=str, default="qwen2.5:0.5b")
    args = parser.parse_args()
    return args

class Baseline:
    def summarize(self, document: str, language: str):
        pass

class GHIC(Baseline):

    def __init__(self, model_name: str):
        embedding_model = Text2EmbeddingsSetenceTransforms(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        spacy_model = 'es_core_news_sm'
        self.document_clusterer = DocumentClustererTopKSentences(embedding_model, spacy_model, top_k_sents=1)

    def summarize(self, document: str, language: str):
        summary = self.document_clusterer.cluster_and_assign(document)[0]
        return summary
    
class ExtractiveSummarizer(Baseline):

    def __init__(self, model_name: str, ratio: int = 0.3):
        custom_config = AutoConfig.from_pretrained(model_name)
        custom_config.output_hidden_states=True
        custom_tokenizer = AutoTokenizer.from_pretrained(model_name)
        custom_model = AutoModel.from_pretrained(model_name, config=custom_config)
        self.model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
        self.ratio = ratio

    def summarize(self, document: str, language: str):
        summary = self.model(document, ratio=self.ratio)
        return summary
    
class OpenAiSummarizer(Baseline):
    def __init__(self, model="gpt-4o-mini", type_sumarization="large"):
        import json
        with open('api/key.json') as f:
            data = json.load(f)
        self.model = model
        self.type_sumarization = type_sumarization
        self.client = OpenAI(
            api_key=data['key'],  # This is the default and can be omitted
        )

    def _generate_prompt(self, document, language):
        """
        Generates a prompt for the summarization task.

        Args:
            document (str): The document to be summarized.

        Returns:
            str: The formatted prompt.
        """
        prompt = f"""Please provide a summary of the following text:

Text: "{document}"

Summary requirements:
1. Length: {self.type_sumarization}.
2. Style: Neutral.
3. Language: {language}.
4. Focus: Main ideas and key points.
"""
        return prompt

    def summarize(self, document, language):
        """
        Summarizes the given document using the OpenAI API.

        Args:
            document (str): The document to be summarized.

        Returns:
            str: The generated summary.
        """
        prompt = self._generate_prompt(document, language)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a model trained to generate institutional summaries of parliamentary minutes. The summaries should be written in formal-administrative language, without value judgments, and follow a clear structure."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract the summary from the response
        summary = response.choices[0].message.content
        return summary
    
class OllamaSummarizer(OpenAiSummarizer):
    def __init__(self, model: str = "llama3", type_sumarization: str = "large"):
        print(f"Using model for Ollama: {model}")
        self.llm = OllamaLLM(model=model)
        self.type_sumarization = type_sumarization

    def summarize(self, document, language):
        prompt = self._generate_prompt(document, language)
        response = self.llm.invoke(prompt)
        return response
    
from summa import summarizer as TextRankSummarizer_model
class TextRankingSummarizer(Baseline):
    
    def __init__(self, model_name: str, ratio: int = 0.3):
        self.ratio = ratio

    def summarize(self, document: str, language: str):
        summary = TextRankSummarizer_model(document,  language=language, ratio=self.ratio)
        return summary
    
    
def generate_summaries(dataset: Dataset, baseline_method: Baseline, num_samples: int=5) -> pd.DataFrame:
        summaries = []
        # get a subset of the dataset
        if num_samples:
            shuffle_dataset = dataset.shuffle(seed=42).select(range(num_samples))
        else:
            shuffle_dataset = dataset
        for obj in tqdm(shuffle_dataset, desc="Generating summaries"):
            input, output, language = obj['input'], obj['output'], obj['language']
            try:
                start_time = time.time()
                summary = baseline_method.summarize(input, language)
                end_time = time.time()
                summaries.append({
                    'document': input, 
                    'expected_summary': output,
                    'generated_summary': summary,
                    'language': language,
                    'time': end_time - start_time,
                })
            except Exception as e:
                print(f"Error: {e}")
                continue

        return pd.DataFrame(summaries)

methods = {
    "ghic": GHIC,
    "openai": OpenAiSummarizer,
    "ollama": OllamaSummarizer,
    "textranking": TextRankingSummarizer,
}

def save_result_baseline(df_summary, method, model_name, name_df):
    if method == "openai" or method == "ghic":
        root = f"models/baseline/{name_df}/{method}"
    else:
        root = f"models/baseline/{name_df}/{method}/{model_name}"
    os.makedirs(root, exist_ok=True)
    df_summary.to_excel(f"{root}/test_summary_normal.xlsx", index=False)
    print("Summaries generated")

def main():
    args = parse()
    print(f"Using method: {args.method}")
    dataset = load_from_disk(args.dataset)
    name_df = args.dataset.split("/")[-1]
    if args.method == "extractive":
        baseline = ExtractiveSummarizer(args.model_name)
    else:
        baseline = methods[args.method](args.model_name)
    # get a subset of the dataset
    subset = dataset["test"].shuffle(seed=42).select(range(20))
    summaries = generate_summaries(subset, baseline, num_samples=None)
    save_result_baseline(summaries, args.method, args.model_name, name_df)

if __name__ == '__main__':
    main()

