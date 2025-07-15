import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import tqdm
import matplotlib.pyplot as plt
from adjustText import adjust_text  # <-- debes instalar esto: pip install adjustText


#delete warnings
import warnings
warnings.filterwarnings("ignore")


def get_model_name_from_path(folder_path: str) -> str:
    return "/".join(folder_path.split("/")[3:])


def load_dataframe(folder: str, filenames: list[str]) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    for name in filenames:
        file_path = os.path.join(folder, name)
        if os.path.exists(file_path):
            return pd.read_excel(file_path, sheet_name="Sheet1"), file_path
    return None, None


def compute_token_lengths(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    df = df[df["generated_summary"].notna() & (df["generated_summary"] != "")]
    df["input_len"] = df["generated_summary"].apply(lambda x: len(tokenizer(x).input_ids))
    return df


def load_bertscore(bertscore_path: str) -> float | None:
    with open(bertscore_path, "r") as f:
        data = json.load(f)
    return data.get("canario", {}).get("bertscore", {}).get("bertscore_f1", None)


def build_metrics(model: str, df: pd.DataFrame, bertscore: float) -> dict:
    return {
        "model": model.split("/")[-1],
        "mean_tokens": round(df["input_len"].mean(), 0),
        "mean_time": round(df["time"].mean(), 2),
        "bertscore": round(bertscore * 100, 2) if bertscore is not None else None
    }


def estimate_parameters(model_name: str) -> float:
    for part in model_name.split("/"):
        if "B" in part:
            try:
                return float(part.lower().replace("b", "").replace("-", ""))
            except ValueError:
                continue
    return 1.0  # fallback


def process_all_models(base_path: str) -> pd.DataFrame:
    results = []
    forbidden_models = [
        "unsloth/Llama-3.1-8B-Instruct",
        "unsloth/Qwen3-8B",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    ]
    filenames = ["test_summary_truncate.xlsx", "test_summary_normal.xlsx"]
    bertscore_filename = "result_metrics.json"

    for root, dirs, files in tqdm.tqdm(os.walk(base_path), desc="Procesando modelos", unit="modelo"):
        if any(fname in files for fname in filenames):
            model_name = get_model_name_from_path(root)
            if model_name in forbidden_models:
                continue
            df, file_path = load_dataframe(root, filenames)
            if df is None:
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                print(f"[ERROR] No se pudo cargar tokenizer para {model_name}: {e}")
                continue
            df = compute_token_lengths(df, tokenizer)
            bertscore_path = os.path.join(root, bertscore_filename)
            if not os.path.exists(bertscore_path):
                print(f"[WARNING] No se encontró BERTScore para {model_name}")
                bertscore = None
            else:
                bertscore = load_bertscore(bertscore_path)
            metrics = build_metrics(model_name, df, bertscore)
            metrics["params"] = estimate_parameters(model_name)
            results.append(metrics)

    return pd.DataFrame(results)


def plot_metrics(df: pd.DataFrame, save_path: str = None):
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # Añadir un pequeño jitter en el eje X e Y para separar puntos muy cercanos
    jitter_strength = 0.2
    x_jitter = df["mean_time"] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    y_jitter = df["mean_tokens"] + np.random.uniform(-jitter_strength * 10, jitter_strength * 10, size=len(df))

    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        x_jitter,
        y_jitter,
        s=df["params"] * 80,
        c=df["bertscore"],
        cmap="coolwarm",
        alpha=0.85,
        edgecolors='none'
    )

    # Etiquetas
    from adjustText import adjust_text
    texts = [plt.text(x, y, name, fontsize=10)
             for x, y, name in zip(x_jitter, y_jitter, df["model"])]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # Colorbar y etiquetas
    cbar = plt.colorbar(scatter)
    cbar.set_label("BERTScore (%)", fontsize=12)
    plt.xlabel("Tiempo medio (s)", fontsize=13)
    plt.ylabel("Longitud media del resumen (tokens)", fontsize=13)
    plt.title("Comparación de modelos: tamaño, tiempo, tokens y calidad (BERTScore)", fontsize=15)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Gráfica guardada en {save_path}")
    plt.show()


if __name__ == "__main__":
    base_models_path = "models/others/data_02-processed_canario"
    df_all_metrics = process_all_models(base_models_path)

    # Guardar CSV
    output_file = os.path.join("metrics", "data", "canario_metrics_time.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_all_metrics.to_csv(output_file, index=False)
    print(f"[INFO] Métricas guardadas en {output_file}")

    # Generar gráfica
    plot_metrics(df_all_metrics, save_path="metrics/data/canario_metrics_plot.png")



