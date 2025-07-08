import os
import time
from datasets import load_from_disk
import pandas as pd
from summarizer import Summarizer
from tqdm import tqdm
from openai import OpenAI
from datasets import Dataset
from langchain_ollama import OllamaLLM, ChatOllama
from transformers import AutoConfig, AutoTokenizer, AutoModel


from document_cluster import DocumentClustererTopKSentences
from text2embeddings import Text2EmbeddingsSetenceTransforms

TOKENIZER_FOR_MODELS = {
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi4": "microsoft/phi-4",
    "qwen3:14b": "Qwen/Qwen3-14B",
    "qwen3:30b" : "Qwen/Qwen3-30B",
}

def parse():
    import argparse
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--dataset', type=str, default="data/02-processed/spanish")
    parser.add_argument('--method', type=str, default="ollama")
    parser.add_argument('--model_name', type=str, default="qwen2.5:0.5b")
    parser.add_argument('--context_window', type=int, default=16384, help='Context window size for the model')
    parser.add_argument('--truncate', type=bool, default=False, help='Whether to truncate the input text to fit the model context window')
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
        prompt = f"""Write an institutional summary in {language} of the following document. Keep the language objective, focusing on facts and agreements:
        "{document}"
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
        system_prompt = SYSTEM_PROMPT[language]
        prompt = INSTRUCTION_TEMPLATE[language]
        prompt = prompt + document + "\n"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract the summary from the response
        summary = response.choices[0].message.content
        return summary

from utils import SYSTEM_PROMPT, INSTRUCTION_TEMPLATE
class OllamaSummarizer(OpenAiSummarizer):
    def __init__(self, model: str = "llama3", type_sumarization: str = "large"):
        print(f"Using model for Ollama: {model}")
        self.model = model
        self.llm = ChatOllama(
            model=model,
            temperature=0.7,  # Adjust temperature for more or less randomness
            max_tokens=2048,  # Adjust max tokens based on your needs
            top_p=0.8,  # Adjust top_p for nucleus sampling
            top_k=20,  # Adjust top_k for top-k sampling
            repetition_penalty=1.0,  # Adjust repetition penalty to avoid repetition
            seed=123,  # For reproducibility
        )
        self.type_sumarization = type_sumarization
        self.qwens = ["qwen3:14b", "qwen3:30b"]

    def summarize(self, document, language):
        system_prompt = SYSTEM_PROMPT[language]
        prompt = INSTRUCTION_TEMPLATE[language]
        prompt = prompt + document + "\n"
        messages = [  
            ("system", system_prompt),  
            ("human", prompt),  
        ]  
        if self.model in self.qwens:
            prompt += f"/no_think"
        response = self.llm.invoke(messages).content
        if self.model in self.qwens:
            response = response.split("</think>")[1].strip()
        return response
    
from summa import summarizer as TextRankSummarizer_model
class TextRankingSummarizer(Baseline):
    
    def __init__(self, model_name: str, ratio: int = 0.3):
        self.ratio = ratio

    def summarize(self, document: str, language: str):
        if language == "canario":
            language = "spanish"
        summary = TextRankSummarizer_model.summarize(document,  language=language, ratio=self.ratio)
        return summary
    

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
nltk.download('punkt_tab')
class LSASummarizer(Baseline):
    
    def __init__(self, model_name: str, ratio: int = 10):
        self.ratio = ratio

    def summarize(self, document: str, language: str):
        if language == "canario":
            language = "spanish"
        parser = PlaintextParser.from_string(document, Tokenizer(language))
        summarizer = LsaSummarizer(Stemmer(language))
        summarizer.stop_words = get_stop_words(language)
        summary = summarizer(parser.document, self.ratio)
        return ' '.join(str(sentence) for sentence in summary)
    
    
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
    "lsa": LSASummarizer,
}

def save_result_baseline(df_summary, method, model_name, name_df, name_excel):
    if method == "openai" or method == "ghic":
        root = f"models/baseline/{name_df}/{method}"
    else:
        root = f"models/baseline/{name_df}/{method}/{model_name}"
    os.makedirs(root, exist_ok=True)
    df_summary.to_excel(f"{root}/{name_excel}", index=False)
    print("Summaries generated")

def truncate_text(text, tokenizer, max_tokens):
    # Tokeniza con truncamiento
    tokenized = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        return_tensors=None,  # para que devuelva dict con input_ids
        return_attention_mask=False,
        return_token_type_ids=False
    )
    # Decodifica para volver a obtener el texto
    truncated_text = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
    return truncated_text

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
    # get tokenizer for the models 
    tokenizer_name = TOKENIZER_FOR_MODELS.get(args.model_name, args.model_name)
    if not tokenizer_name:
        raise ValueError(f"Tokenizer not found for model '{args.model_name}'. Please specify a valid model name.")
    print(f"Using tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def count_tokens_in_dataset(example):
        return {"num_tokens": len(tokenizer(example["input"], add_special_tokens=False)["input_ids"])}
    dataset["test"] = dataset["test"].map(count_tokens_in_dataset)
    if not args.truncate:
        target_tokens = args.context_window - 2049
        data_for_testing = dataset["test"].filter(lambda x: x["num_tokens"] <= target_tokens)
        print(f"Filtering dataset to fit the context window: {target_tokens} tokens")
    else:
        if args.method in ["ghic", "extractive"]:
            data_for_testing = dataset["test"]
            print("No truncation needed for this method")
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            target_tokens = args.context_window - 2049
            def truncate_text_in_dataset(example):
                example["input"] = truncate_text(example["input"], tokenizer, target_tokens)
                return example
            data_for_testing = dataset["test"].map(truncate_text_in_dataset, remove_columns=["num_tokens"])
            print("Truncating text to fit the context window")
    summaries = generate_summaries(data_for_testing, baseline, num_samples=None)
    name_excel = "test_summary_normal.xlsx" if not args.truncate else "test_summary_truncate.xlsx"
    save_result_baseline(summaries, args.method, args.model_name, name_df, name_excel)

if __name__ == '__main__':
    main()

