from collections import defaultdict
import json
import os
import argparse
from time import sleep
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
import wandb
import pandas as pd
from distutils.util import strtobool
from evaluation.summary_metrics_calculator import SummaryMetricsCalculator
from evaluation.document_summary_openai_evaluator import DocumentSummaryOpenAiEvaluator
from utils import (
    DATASET_FILENAME,
    PROJECT_NAME,
    RESULTS_FILENAME,
    seed_everything,
    SEED,
    calculate_weighted_mean,
)

###############################################################################
# Utility helpers
###############################################################################

def load_dataset(model_path: str, filename: str) -> pd.DataFrame:
    """Load the dataset (Excel) located in *model_path* / *filename*."""
    return pd.read_excel(os.path.join(model_path, filename))


def save_metrics_to_json(metrics: dict, model_path: str, filename: str) -> None:
    """Persist *metrics* as pretty‑printed JSON in *model_path* / *filename*."""
    filepath = os.path.join(model_path, filename)
    with open(filepath, "w", encoding="utf‑8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def log_metrics_to_wandb(metrics: dict, use_openai: bool) -> None:
    """Log *metrics* in Weights & Biases using a lang/metric hierarchy."""
    for lang, vals in metrics.items():
        wandb.log({f"{lang}/rouge": vals["rouge"]})
        wandb.log({f"{lang}/bertscore": vals["bertscore"]})
        if use_openai:
            for key in ("coherence", "consistency", "fluency", "relevance", "average"):
                wandb.log({f"{lang}/{key}": vals[key]})
    wandb.finish()


def load_previous_metrics(model_path: str, filename: str) -> dict | None:
    """Return the previously saved metrics file if it exists, otherwise *None*."""
    filepath = os.path.join(model_path, filename)
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf‑8") as f:
            return json.load(f)
    return None

###############################################################################
# Main evaluation pipeline
###############################################################################

def main(
    model: str,
    enable_wandb: bool,
    dataset_hf,
    verbose: bool = True,
    method: str = "normal",
    use_openai: bool = False,
    recalcule_rouge: bool = False,
    up: bool = False,
):

    # ---------------------------------------------------------------------
    # 0)  Determine the output filename and see whether we already have ROUGE
    #     and BERTScore results cached for this model.
    # ---------------------------------------------------------------------
    file_name_to_save = (
        f"{method}_" + RESULTS_FILENAME if method == "truncate" else RESULTS_FILENAME
    )

    previous_metrics = load_previous_metrics(model, file_name_to_save)
    rouge_bertscore_cached = previous_metrics is not None  # already computed once

    # ------------------------------------------------------------------
    # 1)  Set‑up evaluators
    # ------------------------------------------------------------------


    with open("api/key.json", "r", encoding="utf‑8") as fh:
        api_key = json.load(fh)["key"]
    openai_evaluator = DocumentSummaryOpenAiEvaluator(api_key, upgrade=up)

    # ------------------------------------------------------------------
    # 2)  Initialise Weights & Biases if requested
    # ------------------------------------------------------------------
    if enable_wandb:
        wandb.init(
            project=f"{PROJECT_NAME}_metrics",
            entity="miguel_kjh",
            name=os.path.basename(model),
            resume="allow",
        )

    # ------------------------------------------------------------------
    # 3)  Read dataset produced by the model
    # ------------------------------------------------------------------
    name_dataset = f"{DATASET_FILENAME}_{method}.xlsx"
    dataset = load_dataset(model, name_dataset).dropna(
        subset=["expected_summary", "generated_summary"]
    )

    # ------------------------------------------------------------------
    # 4)  Iterate by language and compute / reuse metrics
    # ------------------------------------------------------------------
    metrics: dict[str, dict] = defaultdict(dict)

    for lang, df_lang in dataset.groupby("language"):
        if (rouge_bertscore_cached and lang in previous_metrics) and not recalcule_rouge:
            # ------------------------------------------------------------------
            # We already evaluated ROUGE + BERTScore – reuse those numbers.
            # ------------------------------------------------------------------
            print(f"Reusing ROUGE/BERTScore for {lang} from previous metrics.")
            metrics[lang].update(
                {
                    "rouge": previous_metrics[lang]["rouge"],
                    "bertscore": previous_metrics[lang]["bertscore"],
                    "times(sec)": previous_metrics[lang].get("times(sec)", "N/A"),
                }
            )
            print(f"  metrics[lang]: {metrics[lang]}")
        else:
            # ------------------------------------------------------------------
            # First time evaluating this model: compute ROUGE + BERTScore.
            # ------------------------------------------------------------------
            print(f"Calculating ROUGE/BERTScore for {lang}...")
            calculator = SummaryMetricsCalculator()


            results = calculator.calculate_metrics(
                reference_summaries=df_lang["expected_summary"].tolist(),
                generated_summaries=df_lang["generated_summary"].tolist(),
            )

            times = df_lang["time"].to_numpy()
            metrics[lang]["times(sec)"] = f"{np.mean(times):.2f} ± {np.std(times, ddof=1):.2f}"
            metrics[lang]["rouge"] = results["rouge"]
            metrics[lang]["bertscore"] = results["bertscore"]
            print(f"  metrics[lang]: {metrics[lang]}")

        # ------------------------------------------------------------------
        # 4b)  Always recompute the OpenAI‑based metrics (the expensive part).
        # ------------------------------------------------------------------
        if use_openai:
            openai_scores = {
                "coherence": [],
                "consistency": [],
                "fluency": [],
                "relevance": [],
                "average": [],
            }

            for _, row in tqdm(df_lang.iterrows(), total=len(df_lang), desc=f"OpenAI {lang}"):
                try:
                    input_match = dataset_hf["test"].filter(
                        lambda x: row["expected_summary"] == x["output"]
                    )
                    openai_res = openai_evaluator.evaluate(
                        input_match["input"][0], row["generated_summary"]
                    )
                    # guard against occasional negative values
                    openai_res = {k: max(v, 0) for k, v in openai_res.items()}
                    openai_scores["coherence"].append(openai_res["coherence"])
                    openai_scores["consistency"].append(openai_res["consistency"])
                    openai_scores["fluency"].append(min(openai_res["fluency"], 3))
                    openai_scores["relevance"].append(openai_res["relevance"])
                    openai_scores["average"].append(calculate_weighted_mean(openai_res))
                except Exception as exc:
                    print(f"⚠️  Error evaluating {lang}: {exc}")
                    #sleep(5)  # avoid hitting the OpenAI API too hard

            #sleep(2)  # avoid hitting the OpenAI API too hard
            # Write averaged OpenAI metrics (overwrite if present)
            metrics[lang]["coherence"] = float(np.mean(openai_scores["coherence"]))
            metrics[lang]["consistency"] = float(np.mean(openai_scores["consistency"]))
            metrics[lang]["fluency"] = float(np.mean(openai_scores["fluency"]))
            metrics[lang]["relevance"] = float(np.mean(openai_scores["relevance"]))
            metrics[lang]["average"] = float(np.mean(openai_scores["average"]))

        # ------------------------------------------------------------------
        # 4c)  Optional CLI output for the user
        # ------------------------------------------------------------------
        if verbose:
            print(f"\n── Results: {lang} ───────────────────────────────────────────")
            for key, val in metrics[lang].items():
                print(f"{key:12}: {val}")

    # ------------------------------------------------------------------
    # 5)  Save combined metrics (ROUGE/BERTScore reused, OpenAI fresh)
    # ------------------------------------------------------------------
    save_metrics_to_json(metrics, model, file_name_to_save)

    # ------------------------------------------------------------------
    # 6)  WandB logging if enabled
    # ------------------------------------------------------------------
    if enable_wandb:
        log_metrics_to_wandb(metrics, use_openai)

###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    seed_everything(SEED)

    parser = argparse.ArgumentParser(description="Evaluate model‑generated summaries.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="models/BSC-LT/salamandra-2b-instruct/english/lora/salamandra-2b-instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-02-41-48",
        help="Directory containing the evaluation spreadsheet and metrics JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/02-processed/english",
        help="🤗  Dataset (disk) holding the raw test examples",
    )
    parser.add_argument(
        "--wandb",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Log the metrics to W&B (true/false)",
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Print per‑language breakdown to stdout",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="truncate",
        choices=["normal", "truncate"],
        help="Post‑processing method used when the summaries were generated",
    )
    parser.add_argument(
        "--use_openai",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="(Slow) Evaluate with GPT‑4 rubric as well",
    )
    parser.add_argument(
        "--up",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Use the upgraded OpenAI rubric when available",
    )
    parser.add_argument(
        "--recalcule_rouge",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Recalculate ROUGE and BERTScore even if they are already cached",
    )

    args = parser.parse_args()
    print(f"Evaluating model: {args.model_name_or_path}")
    dataset_hf = load_from_disk(args.dataset)
    main(
        model=args.model_name_or_path,
        enable_wandb=args.wandb,
        dataset_hf=dataset_hf,
        verbose=args.verbose,
        method=args.method,
        use_openai=args.use_openai,
        recalcule_rouge=args.recalcule_rouge,
        up=args.up,
    )
