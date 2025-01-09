import numpy as np
from evaluate import load
from bert_score import BERTScorer
from typing import List

class SummaryMetricsCalculator:
    def __init__(self):
        
        self.rouge = load("rouge")
        self.bertscore = BERTScorer(model_type='bert-base-multilingual-cased')

    def calculate_metrics(self, reference_summaries: List[str], generated_summaries: List[str]):
        """
        Calculates ROUGE and BERTScore metrics for the given lists of summaries.
        :param reference_summaries: List of reference summaries.
        :param generated_summaries: List of generated summaries.
        :return: Dictionary containing ROUGE and BERTScore results.
        """
        if len(reference_summaries) != len(generated_summaries):
            raise ValueError("The lists of summaries must have the same length.")
        
        # Calculate ROUGE metrics
        rouge_results = self.rouge.compute(
            predictions=generated_summaries,
            references=reference_summaries
        )
        
        # Calculate BERTScore metrics
        P, R, F1 = self.bertscore.score(generated_summaries, reference_summaries)

        # Calculate average precision, recall, and F1 using numpy
        bertscore_results = {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }

        # Return results
        return {
            "rouge": rouge_results,
            "bertscore": bertscore_results
        }

# Example usage
if __name__ == "__main__":
    reference_summaries = [
        "The quick brown fox jumps over the lazy dog.", "Hello, world!"
    ]
    generated_summaries = [
        "The brown fox quickly jumped over the slow dog.", "Hello, planet!"
    ]

    calculator = SummaryMetricsCalculator()
    results = calculator.calculate_metrics(reference_summaries, generated_summaries)
    print("ROUGE Results:", results["rouge"])
    print("BERTScore Results:", results["bertscore"])

