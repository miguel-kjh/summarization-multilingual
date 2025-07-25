import os
import json
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import tqdm
import matplotlib.pyplot as plt
from adjustText import adjust_text  # <-- debes instalar esto: pip install adjustText
from nltk.tokenize.toktok import ToktokTokenizer
import string

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import itertools

tokenizer = ToktokTokenizer()
punctuation = set(string.punctuation)  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
LANG = "english"  # Puedes cambiar esto a "spanish", "french", etc.
USE_FINETUNED = True  # Cambia a False si no quieres usar modelos finetuneados
MAX_ = 2000

#delete warnings
import warnings
warnings.filterwarnings("ignore")

models_finetuned = {
    "canario": [
        "models/BSC-LT/salamandra-2b/canario/lora/salamandra-2b-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-01-50-39",
        "models/BSC-LT/salamandra-2b-instruct/canario/lora/salamandra-2b-instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-04-18-33",
        "models/Qwen/Qwen2.5-0.5B/canario/lora/Qwen2.5-0.5B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-03-14-31",
        "models/Qwen/Qwen2.5-0.5B-Instruct/canario/lora/Qwen2.5-0.5B-Instruct-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-14-24-03",
        "models/Qwen/Qwen2.5-1.5B/canario/lora/Qwen2.5-1.5B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-20-18-15",
        "models/Qwen/Qwen2.5-1.5B-Instruct/canario/lora/Qwen2.5-1.5B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-20-23-21",
        "models/Qwen/Qwen2.5-3B/canario/lora/Qwen2.5-3B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-05-34-26",
        "models/Qwen/Qwen2.5-3B-Instruct/canario/lora/Qwen2.5-3B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-09-50-52",
        "models/Qwen/Qwen3-0.6B/canario/lora/Qwen3-0.6B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-05-33-26",
        "models/Qwen/Qwen3-0.6B-Base/canario/lora/Qwen3-0.6B-Base-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-10-22-11",
        "models/Qwen/Qwen3-1.7B/canario/lora/Qwen3-1.7B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-20-59-42",
        "models/Qwen/Qwen3-1.7B-Base/canario/lora/Qwen3-1.7B-Base-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-17-15-40",
        "models/Qwen/Qwen3-4B/canario/lora/Qwen3-4B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-11-03-09",
        "models/Qwen/Qwen3-4B-Base/canario/lora/Qwen3-4B-Base-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-07-54-06",
        "models/unsloth/Llama-3.2-1B/canario/lora/Llama-3.2-1B-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-12-22-29",
        "models/unsloth/Llama-3.2-1B-Instruct/canario/lora/Llama-3.2-1B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-17-37-13",
        "models/unsloth/Llama-3.2-3B/canario/lora/Llama-3.2-3B-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-18-28-10",
        "models/unsloth/Llama-3.2-3B-Instruct/canario/lora/Llama-3.2-3B-Instruct-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-01-15-08",
    ], 
    "english": [
        "models/BSC-LT/salamandra-2b/english/lora/salamandra-2b-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-23-58-11",
        "models/BSC-LT/salamandra-2b-instruct/english/lora/salamandra-2b-instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-02-41-48",
        "models/Qwen/Qwen2.5-0.5B/english/lora/Qwen2.5-0.5B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-22-25-34",
        "models/Qwen/Qwen2.5-0.5B-Instruct/english/lora/Qwen2.5-0.5B-Instruct-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-13-47-05",
        "models/Qwen/Qwen2.5-1.5B/english/lora/Qwen2.5-1.5B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-18-37-59",
        "models/Qwen/Qwen2.5-1.5B-Instruct/english/lora/Qwen2.5-1.5B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-19-00-52",
        "models/Qwen/Qwen2.5-3B/english/lora/Qwen2.5-3B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-02-26-31",
        "models/Qwen/Qwen2.5-3B-Instruct/english/lora/Qwen2.5-3B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-02-46-25",
        "models/Qwen/Qwen3-0.6B/english/lora/Qwen3-0.6B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-04-13-10",
        "models/Qwen/Qwen3-0.6B-Base/english/lora/Qwen3-0.6B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-09-02-27",
        "models/Qwen/Qwen3-1.7B/english/lora/Qwen3-1.7B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-19-16-53",
        "models/Qwen/Qwen3-1.7B-Base/english/lora/Qwen3-1.7B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-15-16-41",
        "models/Qwen/Qwen3-4B/english/lora/Qwen3-4B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-04-08-35",
        "models/Qwen/Qwen3-4B-Base/english/lora/Qwen3-4B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-04-05-15",
        "models/unsloth/Llama-3.2-1B/english/lora/Llama-3.2-1B-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-11-20-00",
        "models/unsloth/Llama-3.2-1B-Instruct/english/lora/Llama-3.2-1B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-16-28-29",
        "models/unsloth/Llama-3.2-3B/english/lora/Llama-3.2-3B-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-16-48-08",
        "models/unsloth/Llama-3.2-3B-Instruct/english/lora/Llama-3.2-3B-Instruct-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-23-45-25",
    ],

}


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
    df["input_len"] = df["generated_summary"].apply(
        lambda x: len([t for t in tokenizer.tokenize(x) if t not in punctuation])
    )
    return df


def load_bertscore(bertscore_path: str) -> float | None:
    with open(bertscore_path, "r") as f:
        data = json.load(f)
    print(f"[INFO] Cargando BERTScore desde {bertscore_path}")
    print(data)
    return data.get(LANG, {}).get("bertscore", {}).get("bertscore_f1", None)


def build_metrics(model: str, df: pd.DataFrame, bertscore: float) -> dict:
    return {
        "model": model.split("/")[-1],
        "mean_tokens": round(df["input_len"].mean(), 0),
        "mean_time": round(df["time"].mean(), 2),
        "bertscore": round(bertscore * 100, 2) if bertscore is not None else None
    }


def estimate_parameters(model_name: str) -> float:
    """
    Extrae el número de parámetros en miles de millones (B) desde el nombre del modelo.
    Ejemplos: 'Qwen2.5-7B', 'LLaMA/3B', 'X/Y/Z/Qwen3-5B-Instruct' → 7.0, 3.0, 5.0, etc.
    """
    match = re.search(r"(\d+\.?\d*)[ -]?B", model_name, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 1.0  # valor por defecto si no se encuentra


def process_all_models(base_path: str) -> pd.DataFrame:
    results = []
    forbidden_models = [
        "unsloth/Llama-3.1-8B-Instruct",
        "unsloth/Qwen3-8B",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    ]
    filenames = ["test_summary_truncate.xlsx", "test_summary_normal.xlsx"]
    bertscore_filename = "truncate_result_metrics.json" 

    generator = None
    if USE_FINETUNED:
        generator = models_finetuned.get(LANG, [])
        for model_path in generator:
            model_name = "/".join(model_path.split("/")[1:3])
            df, file_path = load_dataframe(model_path, filenames)
            if df is None:
                raise FileNotFoundError(f"No se encontraron archivos de métricas en {model_path}")
            print(f"[INFO] samples {len(df)} for {model_name} in {file_path}")
            df = compute_token_lengths(df, tokenizer)
            bertscore_path = os.path.join(model_path, bertscore_filename)
            if not os.path.exists(bertscore_path):
                print(f"[WARNING] No se encontró BERTScore para {model_name}")
                bertscore = None
            else:
                bertscore = load_bertscore(bertscore_path)
            metrics = build_metrics(model_name, df, bertscore)
            metrics["params"] = estimate_parameters(model_name)
            results.append(metrics)
    else:
        generator = os.walk(base_path)

        for root, dirs, files in tqdm.tqdm(generator, desc="Procesando modelos", unit="modelo"):
            if any(fname in files for fname in filenames):
                model_name = get_model_name_from_path(root)
                if model_name in forbidden_models:
                    continue
                df, file_path = load_dataframe(root, filenames)
                if df is None:
                    raise FileNotFoundError(f"No se encontraron archivos de métricas en {root}")
                print(f"[INFO] samples {len(df)} for {model_name} in {file_path}")
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
        cmap="viridis",
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

def plot_academic_scatter(df: pd.DataFrame, save_path: str = None):
    # Normalización y paleta fijas
    norm = mcolors.Normalize(vmin=40, vmax=90)
    colormap = cm.viridis
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)

    # Asignar un símbolo único a cada modelo
    available_markers = [
        'o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', 'H', '8', '1', '2', '3', '4', '|'
    ]
    unique_models = sorted(df["model"].unique())
    model_to_marker = {model: available_markers[i % len(available_markers)] for i, model in enumerate(unique_models)}

    # Crear figura y layout
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[10, 0.5], wspace=0.3, hspace=0.3)

    ax_main = fig.add_subplot(gs[0, 0])
    handles = []

    # Dibujar puntos con símbolo fijo por modelo
    for _, row in df.iterrows():
        marker = model_to_marker[row["model"]]
        color = colormap(norm(row["bertscore"]))
        scatter = ax_main.scatter(
            row["mean_time"],
            row["mean_tokens"],
            color=color,
            marker=marker,
            s=100,
            edgecolor='black',
            alpha=0.8,
            label=row["model"]
        )
        handles.append((scatter, row["model"]))

    # Configurar ejes y título
    ax_main.set_xlabel("Mean Times", fontsize=12)
    ax_main.set_ylabel("Longitud media del resumen (palabras)", fontsize=12)
    ax_main.set_title("Comparación de modelos: tiempo vs longitud con calidad (BERTScore)", fontsize=14)
    ax_main.grid(True)

    # Leyenda a la derecha
    handles = sorted(handles, key=lambda x: x[1])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")
    ax_legend.legend(
        [h[0] for h in handles],
        [h[1] for h in handles],
        loc="upper left",
        fontsize=9,
        title="Modelos"
    )


    # Barra de color horizontal
    ax_cbar = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cbar.set_label("BERTScore")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Gráfica guardada en {save_path}")

    plt.show()

def plot_score_vs_length(df):
    # Paleta de colores para modelos (20 colores diferentes)
    cmap = plt.get_cmap("tab20")
    unique_models = sorted(df["model"].unique())

    # Verificar que hay suficientes marcadores y colores
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', 'H', '8', '1', '2', '3', '4', '|', '_', '+']
    if len(unique_models) > len(markers):
        raise ValueError("No hay suficientes marcadores únicos para todos los modelos.")

    model_to_marker = {model: markers[i] for i, model in enumerate(unique_models)}
    model_to_color = {model: cmap(i % 20) for i, model in enumerate(unique_models)}

    # Preparar figura
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.3)

    ax_main = fig.add_subplot(gs[0])
    handles = []

    for _, row in df.iterrows():
        marker = model_to_marker[row["model"]]
        color = model_to_color[row["model"]]
        point = ax_main.scatter(
            row["mean_tokens"],
            row["bertscore"],
            color=color,
            marker=marker,
            s=100,
            edgecolor='black',
            alpha=0.9
        )
        handles.append((point, row["model"]))

    # Ejes y título
    ax_main.set_xlabel("Longitud media del resumen (palabras)", fontsize=12)
    ax_main.set_ylabel("BERTScore (%)", fontsize=12)
    ax_main.set_title("Relación entre longitud del resumen y calidad (BERTScore)", fontsize=14)
    ax_main.grid(True)

    # Leyenda
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis("off")
    handles = sorted(handles, key=lambda x: x[1])
    ax_legend.legend(
        [h[0] for h in handles],
        [h[1] for h in handles],
        loc="upper left",
        fontsize=9,
        title="Modelos"
    )

    plt.tight_layout()
    plt.show()


def plot_histogram_by_score(df, save_path: str = None):
    # Ordenar por BERTScore descendente
    df_sorted = df.sort_values(by="bertscore", ascending=True)

    # Colores en función del BERTScore
    norm = mcolors.Normalize(vmin=40, vmax=85)
    colormap = cm.viridis
    colors = [colormap(norm(score)) for score in df_sorted["bertscore"]]

    # Crear figura
    plt.figure(figsize=(14, 8))
    bars = plt.bar(
        df_sorted["model"],
        df_sorted["mean_tokens"],
        color=colors,
        edgecolor="black"
    )

    # Añadir texto del BERTScore encima de cada barra
    for bar, score in zip(bars, df_sorted["bertscore"]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f"{score:.1f}", ha='center', va='bottom', fontsize=9)

    # Etiquetas y estilo
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("Summary Length", fontsize=12)

    # Colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("BERTScore")
    plt.ylim(0, MAX_)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600, format='eps')
        print(f"[INFO] Gráfica guardada en {save_path}")
    plt.show()

def plot_histogram_by_score_reversed(df, save_path: str = None):
    # Ordenar por BERTScore ascendente
    df_sorted = df.sort_values(by="bertscore", ascending=True)

    # Definir rangos fijos
    bertscore_min, bertscore_max = 0, 90
    tokens_min, tokens_max = 10, 1000  # ajusta este rango según tu dataset

    # Colores en función de la longitud del resumen
    norm = mcolors.Normalize(vmin=tokens_min, vmax=tokens_max)
    colormap = cm.viridis
    colors = [colormap(norm(length)) for length in df_sorted["mean_tokens"]]

    # Crear figura
    plt.figure(figsize=(14, 8))
    bars = plt.bar(
        df_sorted["model"],
        df_sorted["bertscore"],
        color=colors,
        edgecolor="black"
    )

    # Añadir texto del BERTScore encima de cada barra
    for bar, score in zip(bars, df_sorted["bertscore"]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 f"{score:.1f}", ha='center', va='bottom', fontsize=9)

    # Etiquetas y estilo
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("BERTScore", fontsize=12)
    plt.ylim(bertscore_min, bertscore_max)

    # Colorbar para longitud
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Mean Summary Length")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Gráfica guardada en {save_path}")
    plt.show()




if __name__ == "__main__":

    base_models_path = f"models/others/data_02-processed_{LANG}"
    if not USE_FINETUNED:
        output_file = os.path.join("metrics", "data", f"{LANG}_metrics_time.csv")
    else:
        output_file = os.path.join("metrics", "data", f"{LANG}_finetuned_metrics_time.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(output_file):
        df_all_metrics = process_all_models(base_models_path)
    else:
        df_all_metrics = pd.read_csv(output_file)

    print(df_all_metrics.head())

    # Guardar CSV    
    df_all_metrics.to_csv(output_file, index=False)
    print(f"[INFO] Métricas guardadas en {output_file}")

    # Generar gráfica
    if not USE_FINETUNED:
        save_path = f"metrics/data/{LANG}_metrics_plot.eps"
    else:
        save_path = f"metrics/data/{LANG}_finetuned_metrics_plot.eps"
    #plot_academic_scatter(df_all_metrics, save_path=save_path)
    #plot_academic_scatter(df_all_metrics)
    plot_histogram_by_score(df_all_metrics, save_path=save_path)



