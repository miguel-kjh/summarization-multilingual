#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera una tabla LaTeX a partir de resultados "expert-guided" guardados en subcarpetas por modelo.

- Recorre cada subcarpeta dentro de --root_dir
- Busca un Excel (por defecto: eval_results*.xlsx)
- Lee la hoja agregada (por defecto: paper_summary; detecta variantes comunes)
- Toma la fila summary_id == "ALL" (si no existe, usa la primera fila)
- Construye una tabla LaTeX con SOLO Likert (μ ± σ) para Q1..Q5
- Resalta en NEGRITA el mejor modelo por criterio (máxima media Likert)

Salida:
- --out_tex: archivo .tex con una tabla ready-to-paper
- --out_csv (opcional): dump de los valores agregados extraídos (útil para depurar)

Requisitos:
  pip install pandas openpyxl

Uso:
  python make_latex_table_expert_guided.py \
    --root_dir result \
    --out_tex expert_guided_table.tex \
    --out_csv expert_guided_table.csv

Notas LaTeX:
  Asegúrate de incluir en el preámbulo:
    \\usepackage{booktabs}
"""

import argparse
from pathlib import Path
import pandas as pd


Q_IDS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Q_HEADERS = {
    "Q1": "WHO",
    "Q2": "WHAT",
    "Q3": "HOW",
    "Q4": "WHY",
    "Q5": "ORDER",
}

DEFAULT_EXCEL_GLOB = "eval_results*.xlsx"
DEFAULT_SHEET = "paper_summary"


def find_excel_in_dir(model_dir: Path, excel_glob: str) -> Path | None:
    matches = sorted(model_dir.glob(excel_glob))
    return matches[0] if matches else None


def pick_sheet_name(xlsx_path: Path, preferred: str) -> str:
    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names

    if preferred in sheets:
        return preferred

    # common variants / typos
    candidates = [
        "paper_summary",
    ]
    for c in candidates:
        if c in sheets:
            return c

    # last resort: any sheet containing 'paper'
    for s in sheets:
        if "paper" in s.lower():
            return s

    raise ValueError(f"No paper-like sheet found. Sheets={sheets}")


def format_mean_std_latex(mean_val, std_val, bold: bool, decimals: int) -> str:
    if mean_val is None or pd.isna(mean_val):
        return "-"
    if std_val is None or pd.isna(std_val):
        s = f"{float(mean_val):.{decimals}f}"
    else:
        s = f"{float(mean_val):.{decimals}f} $\\pm$ {float(std_val):.{decimals}f}"
    return f"\\textbf{{{s}}}" if bold else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Carpeta raíz con subcarpetas por modelo (p.ej. result/)")
    ap.add_argument("--out_tex", required=True, help="Ruta de salida .tex")
    ap.add_argument("--out_csv", default=None, help="Ruta de salida .csv (opcional, para depuración)")
    ap.add_argument("--excel_glob", default=DEFAULT_EXCEL_GLOB, help=f"Patrón del Excel (default: {DEFAULT_EXCEL_GLOB})")
    ap.add_argument("--sheet_name", default=DEFAULT_SHEET, help=f"Hoja agregada (default: {DEFAULT_SHEET})")
    ap.add_argument("--decimals", type=int, default=2, help="Decimales para Likert (default: 2)")
    ap.add_argument("--caption", default="Expert-guided automatic evaluation (Tier~3). Likert scores are reported as mean $\\pm$ standard deviation across evaluated summaries. Best-performing models per criterion are highlighted in bold.",
                    help="Caption de la tabla LaTeX")
    ap.add_argument("--label", default="tab:expert-guided", help="Label LaTeX") 
    args = ap.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir not found: {root}")

    extracted = []
    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]): 
        xlsx_path = find_excel_in_dir(model_dir, args.excel_glob)
        if xlsx_path is None:
            continue

        try:
            sheet = pick_sheet_name(xlsx_path, args.sheet_name) 
            df = pd.read_excel(xlsx_path, sheet_name=sheet)

            # pick aggregated row
            if "summary_id" in df.columns:
                agg = df[df["summary_id"].astype(str).str.upper() == "ALL"]
                if len(agg) == 0:
                    agg = df.iloc[[0]]
            else:
                agg = df.iloc[[0]]

            row = agg.iloc[0].to_dict()

            out = {
                "model": model_dir.name,
                "excel": xlsx_path.name,
                "sheet": sheet,
            }

            # collect required columns (means and stds)
            for q in Q_IDS:
                out[f"likert_mean_{q}"] = row.get(f"likert_mean_{q}", None)
                out[f"likert_std_{q}"] = row.get(f"likert_std_{q}", None)

            extracted.append(out)

        except Exception as e:
            extracted.append({
                "model": model_dir.name,
                "excel": xlsx_path.name,
                "sheet": "ERROR",
                "error": str(e),
            })

    if not extracted:
        raise RuntimeError(f"No Excel results found under {root} with glob={args.excel_glob}")

    df_all = pd.DataFrame(extracted)

    # Optional CSV dump
    if args.out_csv:
        df_all.to_csv(args.out_csv, index=False)

    # Filter out errors for LaTeX generation
    df_ok = df_all[df_all["sheet"] != "ERROR"].copy()
    if df_ok.empty:
        raise RuntimeError("All rows failed to parse (sheet == ERROR). Check your excels/sheet names.")

    # Determine best model per criterion (max mean)
    best_per_q = {}
    for q in Q_IDS:
        col = f"likert_mean_{q}"
        best_per_q[q] = pd.to_numeric(df_ok[col], errors="coerce").max(skipna=True)

    # Build LaTeX table rows: Model + 5 criteria
    latex_rows = []
    for _, r in df_ok.iterrows():
        row = [r["model"]]
        for q in Q_IDS:
            mean_val = pd.to_numeric(r.get(f"likert_mean_{q}", None), errors="coerce")
            std_val = pd.to_numeric(r.get(f"likert_std_{q}", None), errors="coerce")

            is_best = (not pd.isna(mean_val)) and (mean_val == best_per_q[q])
            row.append(format_mean_std_latex(mean_val, std_val, bold=is_best, decimals=args.decimals))
        latex_rows.append(row)

    # LaTeX boilerplate
    col_spec = "l" + "c" * len(Q_IDS)
    header = "Model & " + " & ".join(Q_HEADERS[q] for q in Q_IDS) + " \\\\"

    tex = []
    tex.append("\\begin{table*}[t]")
    tex.append("\\centering")
    tex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    tex.append("\\toprule")
    tex.append(header)
    tex.append("\\midrule")

    for row in latex_rows:
        tex.append(" & ".join(row) + " \\\\")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append(f"\\caption{{{args.caption}}}")
    tex.append(f"\\label{{{args.label}}}")
    tex.append("\\end{table*}")

    out_tex_path = Path(args.out_tex)
    out_tex_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"OK: wrote {out_tex_path}")
    if args.out_csv:
        print(f"OK: wrote {args.out_csv}")


if __name__ == "__main__":
    main()
