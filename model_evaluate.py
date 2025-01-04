from collections import defaultdict
import json
import os
import argparse
import wandb
import pandas as pd
from distutils.util import strtobool
from evaluation.summary_metrics_calculator import SummaryMetricsCalculator
from utils import DATASET_FILENAME, PROJECT_NAME, RESULTS_FILENAME, seed_everything, SEED

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
    wandb.finish()

def main(model, enable_wandb, verbose=True):
    # Initialize the summary metrics calculator
    calculator = SummaryMetricsCalculator()

    # Initialize wandb if enabled
    if enable_wandb:
        wandb.init(
            project=f"{PROJECT_NAME}_metrics", 
            entity="miguel_kjh", 
            name=os.path.basename(model),
            resume="allow",
        )

    # Load the dataset
    dataset = load_dataset(model, DATASET_FILENAME)

    # split for language
    dataset_gropby_lang = dataset.groupby("language")

    metrics = defaultdict(dict)
    for lang, dataset in dataset_gropby_lang:
        metrics[lang] = {}
        results = calculator.calculate_metrics(
            reference_summaries=dataset["expected_summary"],
            generated_summaries=dataset["generated_summary"],
            lang="es"
        )

        # Display results
        if verbose:
            print(f"Results for {lang}")
            print("ROUGE Results:", results["rouge"])
            print("BERTScore Results:", results["bertscore"])

        metrics[lang]["rouge"] = results["rouge"]
        metrics[lang]["bertscore"] = results["bertscore"]

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

    args = parser.parse_args()
    main(model=args.model_name_or_path, enable_wandb=args.wandb, verbose=args.verbose)
