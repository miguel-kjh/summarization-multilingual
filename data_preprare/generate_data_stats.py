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
    token_counts = [
        len(tokenizer.encode(sample, truncation=True))
        for sample in dataset['text']
    ]
    return {
        "Tokens Average": round(np.mean(token_counts)),
        "Max Tokens": max(token_counts),
        "Min Tokens": min(token_counts),
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
        sample = sample_dataset(dataset['train'], fraction=self.sample_fraction)
        stats = compute_token_stats(self.tokenizer, sample)
        self.stats['Tokens Average'].append(stats['Tokens Average'])
        self.stats['Max Tokens'].append(stats['Max Tokens'])
        self.stats['Min Tokens'].append(stats['Min Tokens'])

    def get_stats(self) -> pd.DataFrame:
        return pd.DataFrame(self.stats)

