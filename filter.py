# --- parámetros que ya tienes ---
import itertools
import os
import pandas as pd
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from typing import Union

def filter_df_by_token_len(
    base_dir: str,
    dataset_path: Union[str, DatasetDict],
    *,
    input_excel: str = "test_summary_truncate.xlsx",
    output_excel: str = "test_summary_normal.xlsx",
    sheet_name: str = "Sheet1",
    split: str = "test",
    tokenizer_name: str = "unsloth/Llama-3.2-3B-Instruct",
    target_tokens: int = 16_384 - 2_048,   # 14 336
) -> pd.DataFrame:
    """
    Lee un Excel con la columna 'expected_summary', busca en el dataset el
    'input' correspondiente, filtra por longitud ≤ target_tokens y guarda
    otro Excel con el resultado en `base_dir/output_excel`.

    Devuelve el DataFrame filtrado.
    """
    # ── 1) Cargar Excel ───────────────────────────────────────────────────────
    excel_path = os.path.join(base_dir, input_excel)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # ── 2) Cargar dataset (objeto o ruta) y tokenizer ────────────────────────
    ds = load_from_disk(dataset_path) if isinstance(dataset_path, str) else dataset_path
    if split not in ds:
        raise ValueError(f"El split '{split}' no existe en el dataset.")
    dtest = ds[split]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # ── 3) Añadir longitud en tokens al split test ───────────────────────────
    def add_token_len(batch):
        batch["input_len"] = [len(tokenizer(t).input_ids) for t in batch["input"]]
        return batch

    dtest = dtest.map(add_token_len, batched=True, batch_size=512, num_proc=1)

    # ── 4) Diccionario lookup: output → input_len (mínimo si se repite) ──────
    lookup = {}
    for out, l in zip(dtest["output"], dtest["input_len"]):
        lookup[out] = min(l, lookup.get(out, l))

    # ── 5) Añadir longitud y filtrar el DataFrame ────────────────────────────
    df["input_len"] = df["expected_summary"].map(lookup)
    df_filtered = df[df["input_len"].notna() & (df["input_len"] <= target_tokens)].reset_index(drop=True)

    # ── 6) Guardar y devolver ────────────────────────────────────────────────
    output_path = os.path.join(base_dir, output_excel)
    df_filtered.to_excel(output_path, index=False)
    print(f"Excel filtrado guardado en: {output_path}")

    return df_filtered


if __name__ == "__main__":
    DATASET_NAMES = [
        ("portuguese", "data/02-processed/portuguese"),
        ("french", "data/02-processed/french") ,
        ("italian", "data/02-processed/italian"),
        ("german", "data/02-processed/german"),
        ("english", "data/02-processed/english"),
        ("spanish", "data/02-processed/spanish"),
        ("canario", "data/02-processed/canario"),
    ]

    MODEL_NAMES = [
        "models/Qwen/Qwen3-4B/spanish/lora/Qwen3-4B-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-08-06-07",
        "models/Qwen/Qwen3-4B/english/lora/Qwen3-4B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-04-08-35",
        "models/Qwen/Qwen3-4B/french/lora/Qwen3-4B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-17-05-56",
        "models/Qwen/Qwen3-4B/german/lora/Qwen3-4B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-01-02-28",
        "models/Qwen/Qwen3-4B/italian/lora/Qwen3-4B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-10-42-20",
        "models/Qwen/Qwen3-4B/portuguese/lora/Qwen3-4B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-21-54-40",
        "models/Qwen/Qwen3-4B-Base/english/lora/Qwen3-4B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-04-05-15",
        "models/Qwen/Qwen3-4B-Base/french/lora/Qwen3-4B-Base-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-21-44-02",
        "models/Qwen/Qwen3-4B-Base/german/lora/Qwen3-4B-Base-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-01-36-50",
        "models/Qwen/Qwen3-4B-Base/italian/lora/Qwen3-4B-Base-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-23-41-59",
        "models/Qwen/Qwen3-4B-Base/portuguese/lora/Qwen3-4B-Base-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-18-14-18",
        "models/Qwen/Qwen3-4B-Base/spanish/lora/Qwen3-4B-Base-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-05-53-08",
        "models/others/data_02-processed_english/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "models/others/data_02-processed_english/unsloth/Llama-3.1-8B-Instruct",
        "models/others/data_02-processed_french/unsloth/Llama-3.1-8B-Instruct",
        "models/others/data_02-processed_french/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "models/others/data_02-processed_german/unsloth/Llama-3.1-8B-Instruct",
        "models/others/data_02-processed_german/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "models/others/data_02-processed_italian/unsloth/Llama-3.1-8B-Instruct",
        "models/others/data_02-processed_italian/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "models/others/data_02-processed_portuguese/unsloth/Llama-3.1-8B-Instruct",
        "models/others/data_02-processed_portuguese/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "models/others/data_02-processed_spanish/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    ]

    # Ejemplo de uso
    for (base_dir, dataset_path) in itertools.product(MODEL_NAMES, DATASET_NAMES):
        lang = dataset_path[0]
        if lang not in base_dir:
            continue
        _ = filter_df_by_token_len(base_dir, dataset_path[1])


