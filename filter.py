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
    tokenizer_name: str = "BSC-LT/salamandra-2b-instruct",  # Cambia esto al nombre del tokenizer que estés usando
    target_tokens: int = 8192 -1300,  # Longitud máxima de tokens
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

    print(f"Filtrado: {len(df_filtered)} filas con longitud de tokens ≤ {target_tokens}")

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
        "models/others/data_02-processed_canario/BSC-LT/salamandra-2b-instruct",
    ]

    # Ejemplo de uso
    for (base_dir, dataset_path) in itertools.product(MODEL_NAMES, DATASET_NAMES):
        lang = dataset_path[0]
        if lang not in base_dir:
            continue
        _ = filter_df_by_token_len(base_dir, dataset_path[1])


