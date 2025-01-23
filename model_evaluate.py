from collections import defaultdict
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
import wandb
import pandas as pd
from distutils.util import strtobool
from evaluation.summary_metrics_calculator import SummaryMetricsCalculator
from evaluation.document_summary_openai_evaluator import DocumentSummaryOpenAiEvaluator
from utils import DATASET_FILENAME, PROJECT_NAME, RESULTS_FILENAME, seed_everything, SEED, calculate_weighted_mean

def load_dataset(model_path, filename):
    """
    Load the dataset from an Excel file.

    :param model_path: Path to the model directory.
    :param filename: Name of the dataset file.
    :return: Loaded dataset as a pandas DataFrame.
    """
    filepath = os.path.join(model_path, filename)
    return pd.read_excel(filepath)

def save_metrics_to_json(metrics, model_path, filename):
    """
    Save the calculated metrics to a JSON file.

    :param metrics: Dictionary containing the metrics.
    :param model_path: Path to the model directory.
    :param filename: Name of the output file.
    """
    filepath = os.path.join(model_path, filename)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)

def log_metrics_to_wandb(metrics: dict):
    """
    Log metrics to Weights & Biases (wandb).

    :param metrics: Dictionary containing the metrics.
    """
    for lang, metrics in metrics.items():
        wandb.log({f"{lang}/rouge": metrics["rouge"]})
        wandb.log({f"{lang}/bertscore": metrics["bertscore"]})
        wandb.log({f"{lang}/coherence": metrics["coherence"]})
        wandb.log({f"{lang}/consistency": metrics["consistency"]})
        wandb.log({f"{lang}/fluency": metrics["fluency"]})
        wandb.log({f"{lang}/relevance": metrics["relevance"]})
        wandb.log({f"{lang}/average": metrics["average"]})
    wandb.finish()

def main(model, enable_wandb, verbose=True, method="normal"):
    # Initialize the summary metrics calculator
    calculator = SummaryMetricsCalculator()
    with open("api/key.json", "r") as file:
        api_key = json.load(file)["key"]
    openai_evaluator = DocumentSummaryOpenAiEvaluator(api_key)

    # Initialize wandb if enabled
    if enable_wandb:
        wandb.init(
            project=f"{PROJECT_NAME}_metrics", 
            entity="miguel_kjh", 
            name=os.path.basename(model),
            resume="allow",
        )

    # Load the dataset
    name_dataset = f"{DATASET_FILENAME}_{method}.xlsx"
    dataset = load_dataset(model, name_dataset)

    # split for language
    dataset_gropby_lang = dataset.groupby("language")

    metrics = defaultdict(dict)
    for lang, dataset in dataset_gropby_lang:
        metrics[lang] = {}
        results = calculator.calculate_metrics(
            reference_summaries=dataset["expected_summary"].to_list(),
            generated_summaries=dataset["generated_summary"].to_list(),
        )

        metrics[lang]["rouge"] = results["rouge"]
        metrics[lang]["bertscore"] = results["bertscore"]

        # OpenAI evaluation
        openai_metrics = {
            'coherence': [],
            'consistency': [],
            'fluency': [], 
            'relevance': [],
            'average': [],
        }
        for _, row in tqdm(dataset.iterrows(), desc=f"Evaluating {lang}"):
            try:
                openai_results = openai_evaluator.evaluate(
                    row["expected_summary"], 
                    row["generated_summary"],
                )
                openai_metrics['coherence'].append(openai_results['coherence'])
                openai_metrics['consistency'].append(openai_results['consistency'])
                openai_metrics['fluency'].append(openai_results['fluency'])
                openai_metrics['relevance'].append(openai_results['relevance'])
                openai_metrics['average'].append(calculate_weighted_mean(openai_results))
            except Exception as e:
                continue
        
        metrics[lang]["coherence"] = np.mean(openai_metrics['coherence'])
        metrics[lang]["consistency"] = np.mean(openai_metrics['consistency'])
        metrics[lang]["fluency"] = np.mean(openai_metrics['fluency'])
        metrics[lang]["relevance"] = np.mean(openai_metrics['relevance'])
        metrics[lang]["average"] = np.mean(openai_metrics['average'])

        # Display results
        if verbose:
            print(f"Results for {lang}")
            print("ROUGE Results:", metrics[lang]["rouge"])
            print("BERTScore Results:", metrics[lang]["bertscore"])
            print("OpenAI Evaluation Results:")
            print("Coherence:", metrics[lang]["coherence"])
            print("Consistency:", metrics[lang]["consistency"])
            print("Fluency:", metrics[lang]["fluency"])
            print("Relevance:", metrics[lang]["relevance"])
            print("Average:", metrics[lang]["average"])

    # Save metrics to a JSON file
    save_metrics_to_json(metrics, model, RESULTS_FILENAME)

    # Log metrics to wandb if enabled
    if enable_wandb:
        log_metrics_to_wandb(metrics)

if __name__ == "__main__":
    seed_everything(SEED)
    parser = argparse.ArgumentParser(description="Evaluate model-generated summaries.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model directory (e.g., 'models/pythia-14m-tiny-e20-b8-lr0.0001-wd0.01-c512-r16-a32-d0.05')."
    )
    parser.add_argument( 
        "--wandb",
        type=lambda x: bool(strtobool(x)), 
        default=False,
        help="Enable logging to Weights & Biases (wandb). Set to True to enable."
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Enable verbose output. Set to True to enable."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="normal",
        help="Method to use for generating summaries. Options: normal, topk, clf."
    )

    args = parser.parse_args()
    assert args.method in ["normal", "topk", "clf"], f"Invalid method: {args.method}"
    main(model=args.model_name_or_path, enable_wandb=args.wandb, verbose=args.verbose, method=args.method)
