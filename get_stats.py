#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Union

import numpy as np
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

from transformers import AutoTokenizer
from nltk.tokenize import ToktokTokenizer
import nltk

# -----------------------------
# Utils
# -----------------------------

def pct(x):
    return np.array(x) * 100.0

def mean(x):
    return float(np.mean(x)) if len(x) else 0.0

def median(x):
    return float(np.median(x)) if len(x) else 0.0

def percentile(x, p):
    return float(np.percentile(x, p)) if len(x) else 0.0

def summarize_vector(x: List[float], prefix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}avg": mean(x),
        f"{prefix}p50": median(x),
        f"{prefix}p90": percentile(x, 90),
        f"{prefix}p95": percentile(x, 95),
    }

# -----------------------------
# Tokenization layers
# -----------------------------

@dataclass
class TokenizedExample:
    # Para la capa que estés usando (word o token); ambas listas son opcionales
    input_seq: List[Union[int, str]]
    summary_seq: List[Union[int, str]]

class TokenizerWrapper:
    def __init__(self, hf_tokenizer=None, lowercase=False, level="token"):
        self.hf_tokenizer = hf_tokenizer
        self.lowercase = lowercase
        self.level = level
        self.word_tokenizer = ToktokTokenizer()

    def _pre(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def tokenize_for_level(self, text: str) -> List[Union[int, str]]:
        text = self._pre(text)
        if self.level == "token":
            # ids del modelo, sin special tokens
            return self.hf_tokenizer(text, add_special_tokens=False).input_ids
        elif self.level == "word":
            return self.word_tokenizer.tokenize(text)
        else:
            raise ValueError(f"Unknown level: {self.level}")

    def tokens_count_model(self, text: str) -> int:
        # para contar tokens del modelo (independiente del nivel de análisis)
        return len(self.hf_tokenizer(text, add_special_tokens=False).input_ids)

# -----------------------------
# N-grams
# -----------------------------

def ngrams_list(seq: List, n: int) -> List[Tuple]:
    if len(seq) < n:
        return []
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

def pct_new_ngrams(summary_seq: List, input_seq: List, n: int, use_frequencies: bool) -> float:
    """
    Si use_frequencies=True, medimos por ocurrencias:
        (# de ocurrencias de n-grams del resumen que NO están en el input) / (# total de ocurrencias de n-grams del resumen)
    Si False, medimos por tipos (sets), como en tu código original.
    """
    s_ngrams = ngrams_list(summary_seq, n)
    if not s_ngrams:
        return 0.0

    if use_frequencies:
        i_set = set(ngrams_list(input_seq, n))
        not_in = sum(1 for g in s_ngrams if g not in i_set)
        return 100.0 * not_in / len(s_ngrams)
    else:
        s_set = set(s_ngrams)
        i_set = set(ngrams_list(input_seq, n))
        new = s_set - i_set
        return 100.0 * len(new) / len(s_set)

# -----------------------------
# Coverage & Density
# -----------------------------

def build_positions_index(a_tokens: List) -> Dict:
    """
    Devuelve un dict token -> lista de posiciones donde aparece en a_tokens.
    Acelera la búsqueda del longest match.
    """
    idx = defaultdict(list)
    for i, tok in enumerate(a_tokens):
        idx[tok].append(i)
    return idx

def longest_match_fragments(a_tokens: List, s_tokens: List) -> List[List]:
    """
    Greedy longest-match como en NEWSROOM:
    Para cada posición i en el resumen, busca el fragmento más largo que esté en el input
    y avanza i por la longitud del match (o 1 si no hay match).
    """
    if not s_tokens:
        return []

    positions = build_positions_index(a_tokens)
    fragments = []
    i = 0
    len_a = len(a_tokens)
    len_s = len(s_tokens)

    while i < len_s:
        tok = s_tokens[i]
        best_k = 0

        # Posibles comienzos en a_tokens para ese token
        candidates = positions.get(tok, [])
        for j in candidates:
            k = 0
            # avanza mientras coincidan
            while (i + k < len_s) and (j + k < len_a) and (s_tokens[i + k] == a_tokens[j + k]):
                k += 1
            if k > best_k:
                best_k = k

        if best_k > 0:
            fragments.append(s_tokens[i:i + best_k])
            i += best_k
        else:
            i += 1

    return fragments

def coverage_density(a_tokens: List, s_tokens: List) -> Tuple[float, float]:
    if not s_tokens:
        return 0.0, 0.0
    fragments = longest_match_fragments(a_tokens, s_tokens)
    frag_lens = [len(f) for f in fragments]
    coverage = sum(frag_lens) / len(s_tokens)
    density = sum(l * l for l in frag_lens) / len(s_tokens)
    return coverage, density

# -----------------------------
# Core computation
# -----------------------------

def compute_statistics(dataset, model_tokenizer, level: str, lowercase: bool,
                       use_freq_ngrams: bool) -> Dict[str, float]:
    """
    Calcula estadísticas a un nivel (word o token).
    """
    tw = TokenizerWrapper(model_tokenizer, lowercase=lowercase, level=level)
    toktok = ToktokTokenizer()

    input_lengths = []
    summary_lengths = []
    # Siempre contamos palabras del resumen para "Summary words avg"
    summary_words_lengths = []

    compression_ratios = []
    coverages = []
    densities = []
    new_ngrams_total = {1: [], 2: [], 3: [], 4: []}

    for sample in tqdm(dataset, desc=f"Computing stats ({level}-level)"):
        try:
            input_text = sample["input"]
            summary_text = sample["output"]
        except KeyError as e:
            input_text = sample["original_text"]
            summary_text = sample["summarized_text"]

        # Secuencias para este nivel
        input_seq = tw.tokenize_for_level(input_text)
        summary_seq = tw.tokenize_for_level(summary_text)

        # Para contar palabras (siempre sobre palabras)
        summary_words = toktok.tokenize(summary_text.lower() if lowercase else summary_text)

        # Para el compression ratio usamos el mismo nivel que estamos midiendo
        # (consistente con tus métricas originales)
        len_a = len(input_seq)
        len_s = len(summary_seq)

        input_lengths.append(len_a)
        summary_lengths.append(len_s)
        summary_words_lengths.append(len(summary_words))
        compression_ratios.append((len_a / len_s) if len_s else 0.0)

        for n in range(1, 5):
            new_pct = pct_new_ngrams(summary_seq, input_seq, n, use_frequencies=use_freq_ngrams)
            new_ngrams_total[n].append(new_pct)

        cov, dens = coverage_density(input_seq, summary_seq)
        coverages.append(cov)
        densities.append(dens)

    # Promedios + percentiles
    results = {}
    results.update(summarize_vector(input_lengths, prefix="Input tokens "))
    results.update(summarize_vector(summary_lengths, prefix="Summary tokens "))
    results.update(summarize_vector(summary_words_lengths, prefix="Summary words "))
    results.update(summarize_vector(compression_ratios, prefix="Compression ratio "))

    results["Coverage avg"] = mean(coverages)
    results["Density avg"] = mean(densities)

    for n in range(1, 5):
        results[f"New {n}-grams (%) avg"] = mean(new_ngrams_total[n])

    return results

# -----------------------------
# Pretty printing
# -----------------------------

def print_block(title: str, stats: Dict[str, float]):
    print(f"\n=== {title} ===")
    for k, v in sorted(stats.items()):
        print(f"{k}: {v:.4f}")

# -----------------------------
# Main
# -----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute summarization stats with improved coverage/density and n-gram novelty.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset (HF load_from_disk)")
    parser.add_argument("--model", type=str, default="qwen", help="Model alias (llama/qwen) or HF path")
    parser.add_argument("--level", type=str, choices=["word", "token", "both"], default="token",
                        help="Granularidad para las métricas principales")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase texts before processing")
    parser.add_argument("--no-freq-ngrams", action="store_true",
                        help="Usar sets (tipos) para % de nuevos n-grams en lugar de ocurrencias")
    parser.add_argument("--save_to", type=str, default=None, help="Ruta del fichero donde guardar resultados (txt)")
    return parser.parse_args()

def resolve_model_path(alias: str) -> str:
    models = {
        "llama": "meta-llama/Llama-3.2-1B",
        "qwen": "Qwen/Qwen3-4B",
    }
    return models.get(alias, alias)

def main():
    nltk.download('punkt', quiet=True)

    args = parse_arguments()
    model_path = resolve_model_path(args.model)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist.")

    print(f"Loading dataset from: {args.dataset_path}")
    data = load_from_disk(args.dataset_path)

    splits = []
    for split_name in ["train", "test", "validation"]:
        if split_name in data:
            splits.append(data[split_name])
    if not splits:
        raise ValueError("No se encontraron splits válidos en el dataset.")

    dataset = concatenate_datasets(splits)
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    use_freq = not args.no_freq_ngrams

    all_stats = {}

    if args.level in ("token", "both"):
        tok_stats = compute_statistics(dataset, tokenizer, level="token",
                                       lowercase=args.lowercase, use_freq_ngrams=use_freq)
        all_stats["TOKEN LEVEL"] = tok_stats

    if args.level in ("word", "both"):
        word_stats = compute_statistics(dataset, tokenizer, level="word",
                                        lowercase=args.lowercase, use_freq_ngrams=use_freq)
        all_stats["WORD LEVEL"] = word_stats

    for title, stats in all_stats.items():
        print_block(title, stats)

    if args.save_to:
        with open(args.save_to, "w", encoding="utf-8") as f:
            for title, stats in all_stats.items():
                f.write(f"=== {title} ===\n")
                for k, v in sorted(stats.items()):
                    f.write(f"{k}: {v:.4f}\n")
        print(f"\nResultados guardados en: {args.save_to}")

if __name__ == "__main__":
    main()
