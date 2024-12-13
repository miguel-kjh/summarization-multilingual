import os
import numpy as np
import pandas as pd
from typing import Dict
from collections import defaultdict
from transformers import AutoTokenizer

from utils import SEED
    
def sample_dataset(dataset, fraction=0.05):
    return dataset.shuffle(seed=SEED).select(range(int(len(dataset) * fraction)))

def compute_token_stats(tokenizer, dataset):
    token_counts_texts = [
        len(tokenizer.encode(sample, truncation=True))
        for sample in dataset['text']
    ]
    token_counts_summaries = [
        len(tokenizer.encode(sample, truncation=True))
        for sample in dataset['summary']
    ]
    return {
        "Tokens Average": round(np.mean(token_counts_texts)),
        "Max Tokens": max(token_counts_texts),
        "Min Tokens": min(token_counts_texts),
        "Tokens Average Summaries": round(np.mean(token_counts_summaries)),
        "Max Tokens Summaries": max(token_counts_summaries),
        "Min Tokens Summaries": min(token_counts_summaries),
    }

class StatsGenerator:

    def __init__(self, sample_fraction=0.01):
        tokenizer_name: str = "BSC-LT/salamandra-7b"
        self.stats = defaultdict(list)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sample_fraction = sample_fraction

    def add_lang(self, dataset: Dict, lang: str):
        self.stats['Lang'].append(lang)
        
        for split, data in dataset.items():
            # Número de muestras
            self.stats[f'Samples {split}'].append(data.num_rows)

        # Muestrear y calcular estadísticas
        #dataset_sampled = sample_dataset(dataset['train'], fraction=self.sample_fraction)
        stats = compute_token_stats(self.tokenizer, dataset['train'])
        self.stats['Tokens Average'].append(stats['Tokens Average'])
        self.stats['Max Tokens'].append(stats['Max Tokens'])
        self.stats['Min Tokens'].append(stats['Min Tokens'])
        self.stats['Tokens Average Summaries'].append(stats['Tokens Average Summaries'])
        self.stats['Max Tokens Summaries'].append(stats['Max Tokens Summaries'])
        self.stats['Min Tokens Summaries'].append(stats['Min Tokens Summaries'])


    def get_stats(self) -> pd.DataFrame:
        return pd.DataFrame(self.stats)

