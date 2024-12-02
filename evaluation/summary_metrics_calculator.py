import numpy as np
from evaluate import load
from typing import List

class SummaryMetricsCalculator:
    def __init__(self):
        """
        Initializes the class with the specified language for BERTScore.
        :param lang: Language to use for BERTScore. Example: 'en', 'es'.
        """
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")

    def calculate_metrics(self, reference_summaries: List[str], generated_summaries: List[str], lang: str="en"):
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
        bertscore_raw = self.bertscore.compute(
            predictions=generated_summaries,
            references=reference_summaries,
            lang=lang
        )

        # Calculate average precision, recall, and F1 using numpy
        bertscore_results = {
            "precision": np.mean(bertscore_raw["precision"]),
            "recall": np.mean(bertscore_raw["recall"]),
            "f1": np.mean(bertscore_raw["f1"]),
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
    results = calculator.calculate_metrics(reference_summaries, generated_summaries, lang="en")
    print("ROUGE Results:", results["rouge"])
    print("BERTScore Results:", results["bertscore"])

