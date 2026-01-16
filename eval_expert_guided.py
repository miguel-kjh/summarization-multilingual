#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from tqdm import tqdm
from typing import Any, Dict

import numpy as np
import pandas as pd
from openai import OpenAI

# ---------------------------
# Prompt + JSON Schema
# ---------------------------

Q1 = "¿El resumen identifica explícitamente a los intervinientes principales, incluyendo su nombre y Grupo Parlamentario cuando procede?"
Q2 = "¿El resumen permite identificar explícitamente uno o más actos ilocutorios realizados durante el debate (p. ej., preguntar, responder, proponer, enmendar, fijar posición, votar, aprobar, señalar, exponer u otros), independientemente del contenido temático tratado?"
Q3 = "¿El resumen describe cómo se abordó el punto del orden del día, indicando el contenido principal discutido en el debate?"
Q4 = "¿El resumen recoge la existencia del parecer o posicionamiento final de los grupos parlamentarios respecto al punto tratado, aunque no detalle los argumentos?"
Q5 = "¿El resumen respeta el orden cronológico de las intervenciones y mantiene coherencia con la secuencia del Diario de Sesiones?"


EVAL_SYSTEM_PROMPT = f"""
Eres un evaluador experto en análisis de debates parlamentarios (Diario de Sesiones) y en teoría de los actos del habla (Austin, Searle).

Tu tarea es evaluar si un RESUMEN permite inferir correctamente la información procedimental y factual relevante, tal y como lo haría un experto humano al resumir un debate parlamentario.

IMPORTANTE (criterios):
- Este tipo de resumen es intencionadamente corto y de carácter procedimental.
- Los resúmenes humanos expertos no describen el contenido temático del debate, sino los actos ilocutorios realizados mediante el lenguaje.
- Debes evaluar qué se hace al hablar, no de qué se habla.
- Ejemplos de actos ilocutorios parlamentarios relevantes incluyen: exponer o fundamentar una iniciativa, informar, responder, preguntar, señalar el parecer o fijar posición de los grupos, replicar, admitir a trámite, votar, aprobar.
- La ausencia de referencias al tema, asunto o contenido del debate es normal y no debe considerarse confusa ni incompleta.
- Prioriza actores, roles, estructura de turnos y acciones procedimentales.
- NO penalices la ausencia de valoraciones, conclusiones políticas o detalles argumentativos extensos.
- El DOCUMENTO_ORIGINAL actúa como ground truth.
- Evalúa únicamente el RESUMEN.
- Si el resumen introduce información no presente en el DOCUMENTO_ORIGINAL, penaliza la puntuación aunque parezca plausible.

Escala Likert (1–5) para cada criterio:
1 = No aparece en el resumen del modelo
2 = Aparece pero es incorrecto o confusa
3 = Aparece parcialmente o incompleto
4 = Aparece de forma clara y mayormente correcta (pequeñas omisiones)
5 = Aparece de forma clara, precisa y completa (sin alucinaciones)

Regla yes/no:
- "yes" si el criterio se cumple razonablemente (≈ Likert >= 4), si no "no"

Para cada pregunta incluye:
- yes_no (yes/no)
- likert (1..5)
- evidence_model: cita breve (máx. 25 palabras) del RESUMEN o "" si no hay evidencia
- notes: explicación corta (1–2 frases)

Preguntas / criterios a evaluar (sobre el RESUMEN):

Q1 (WHO - Intervinientes):
{Q1}

Q2 (WHAT – Speech Acts):
{Q2}

Q3 (HOW - Desarrollo procedimental):
{Q3}

Q4 (WHY - Parecer/posicionamiento):
{Q4}

Q5 (ORDER - Secuencia):
{Q5}
"""


EVAL_USER_TEMPLATE = """Ahora evalúa usando los textos siguientes.

DOCUMENTO_ORIGINAL:
<<<
{ref}
>>>

RESUMEN:
<<<
{hyp}
>>>
"""

#TODO: CAMBIAR LAS QUESTIONES Y EL JSON SCHEMA
EVAL_JSON_SCHEMA = {
    "name": "parliamentary_summary_expert_guided_eval",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "meta": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task": {"type": "string", "const": "expert_guided_eval_parliamentary_summary"},
                    "language": {"type": "string", "const": "es"},
                    "likert_definition": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "1": {"type": "string"},
                            "2": {"type": "string"},
                            "3": {"type": "string"},
                            "4": {"type": "string"},
                            "5": {"type": "string"},
                        },
                        "required": ["1", "2", "3", "4", "5"],
                    },
                },
                "required": ["task", "language", "likert_definition"],
            },

            # Key change: results is an OBJECT with fixed keys, not an array
            "results": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "Q1": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "const": "Q1"},
                            "criterion": {"type": "string", "const": "WHO - Intervinientes"},
                            "question": {
                                "type": "string",
                                "const": Q1,
                            },
                            "yes_no": {"type": "string", "enum": ["yes", "no"]},
                            "likert": {"type": "integer", "minimum": 1, "maximum": 5},
                            "evidence_model": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["id", "criterion", "question", "yes_no", "likert", "evidence_model", "notes"],
                    },
                    "Q2": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "const": "Q2"},
                            "criterion": {"type": "string", "const": "WHAT - Speech Acts"},
                            "question": {
                                "type": "string",
                                "const": Q2,
                            },
                            "yes_no": {"type": "string", "enum": ["yes", "no"]},
                            "likert": {"type": "integer", "minimum": 1, "maximum": 5},
                            "evidence_model": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["id", "criterion", "question", "yes_no", "likert", "evidence_model", "notes"],
                    },
                    "Q3": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "const": "Q3"},
                            "criterion": {"type": "string", "const": "HOW - Desarrollo procedimental"},
                            "question": {
                                "type": "string",
                                "const": Q3,
                            },
                            "yes_no": {"type": "string", "enum": ["yes", "no"]},
                            "likert": {"type": "integer", "minimum": 1, "maximum": 5},
                            "evidence_model": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["id", "criterion", "question", "yes_no", "likert", "evidence_model", "notes"],
                    },
                    "Q4": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "const": "Q4"},
                            "criterion": {"type": "string", "const": "WHY - Parecer/posicionamiento"},
                            "question": {
                                "type": "string",
                                "const": Q4,
                            },
                            "yes_no": {"type": "string", "enum": ["yes", "no"]},
                            "likert": {"type": "integer", "minimum": 1, "maximum": 5},
                            "evidence_model": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["id", "criterion", "question", "yes_no", "likert", "evidence_model", "notes"],
                    },
                    "Q5": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "const": "Q5"},
                            "criterion": {"type": "string", "const": "ORDER - Secuencia"},
                            "question": {
                                "type": "string",
                                "const": Q5,    
                            },
                            "yes_no": {"type": "string", "enum": ["yes", "no"]},
                            "likert": {"type": "integer", "minimum": 1, "maximum": 5},
                            "evidence_model": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["id", "criterion", "question", "yes_no", "likert", "evidence_model", "notes"],
                    },
                },
                "required": ["Q1", "Q2", "Q3", "Q4", "Q5"],
            },

            "overall": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "average_likert": {"type": "number"},
                    "pass_all_yes": {"type": "boolean"},
                    "main_issues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["average_likert", "pass_all_yes", "main_issues"],
            },
        },
        "required": ["meta", "results", "overall"],
    },
}



def call_openai_eval(client: OpenAI, model: str, ref: str, hyp: str, max_retries: int = 3) -> Dict[str, Any]:
    user_prompt = EVAL_USER_TEMPLATE.format(ref=ref, hyp=hyp)
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,  # deterministic
                response_format={"type": "json_schema", "json_schema": EVAL_JSON_SCHEMA},
                seed=123,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response content")
            data = json.loads(content)
            if "results" not in data or len(data["results"]) != 5:
                raise ValueError("Invalid JSON: missing results[5]")
            return data

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.2)
            else:
                raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}") from last_err

    raise RuntimeError(f"OpenAI call failed: {last_err}")


def majority_yes_over_summaries(series: pd.Series) -> str:
    """Global majority across summaries: yes if > 0.5."""
    p_yes = (series == "yes").mean()
    return "yes" if p_yes > 0.5 else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_xlsx", required=True) 
    ap.add_argument("--ref_col", default="document")
    ap.add_argument("--hyp_col", default="generated_summary")
    ap.add_argument("--id_col", default=None, help="Optional ID column; otherwise uses original row index")
    ap.add_argument("--n_summaries", type=int, default=100, help="Number of summaries to sample for evaluation")
    ap.add_argument("--sample_seed", type=int, default=123)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()
    args.input_xlsx = os.path.join(args.input_xlsx, "test_summary_truncate.xlsx")

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(args.input_xlsx)
    for col in [args.ref_col, args.hyp_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input Excel.") 

    df = df.sort_values(by='document').reset_index(drop=True)
    n_pick = min(args.n_summaries, len(df)) if args.n_summaries is not None else len(df)
    sampled = df.sample(n=n_pick, random_state=args.sample_seed).copy()
    sampled = sampled.reset_index(drop=False).rename(columns={"index": "_orig_row"})
    if args.id_col and args.id_col in sampled.columns:
        sampled["summary_id"] = sampled[args.id_col].astype(str)
    else:
        sampled["summary_id"] = sampled["_orig_row"].astype(str)

    client = OpenAI()

    raw_rows = []
    sid = 0
    for _, row in tqdm(sampled.iterrows(), total=len(sampled)): 
        sid += 1
        ref = "" if pd.isna(row[args.ref_col]) else str(row[args.ref_col])
        hyp = "" if pd.isna(row[args.hyp_col]) else str(row[args.hyp_col])

        data = call_openai_eval(client, args.model, ref, hyp)
        for qid, r in data["results"].items():
            raw_rows.append({
                "summary_id": sid,
                "question_id": qid,          # o r["id"] (son iguales)
                "yes_no": r["yes_no"],
                "likert": int(r["likert"]),
                "evidence_model": r.get("evidence_model", ""),
                "notes": r.get("notes", ""),
            })
            #if qid == "Q2":
            #    print(f"DEBUG: summary_id={sid} Q2: {int(r['likert'])} yes_no={r['yes_no']}")
            #    print(f"evidence_model: {r['evidence_model']}")
            #    print(f"notes: {r['notes']}")        

    raw = pd.DataFrame(raw_rows)

    # Wide per-summary
    wide_likert = raw.pivot(index="summary_id", columns="question_id", values="likert").add_prefix("likert_")
    wide_yesno = raw.pivot(index="summary_id", columns="question_id", values="yes_no").add_prefix("yesno_")
    final_wide = pd.concat([wide_likert, wide_yesno], axis=1).reset_index()

    # Paper summary: ONE row, macro-average across summaries
    paper_rows = {"summary_id": "ALL"}
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        q_df = raw[raw["question_id"] == q]
        paper_rows[f"likert_mean_{q}"] = float(q_df["likert"].mean())
        paper_rows[f"likert_std_{q}"] = float(q_df["likert"].std(ddof=1)) if len(q_df) > 1 else 0.0
        paper_rows[f"likert_median_{q}"] = float(q_df["likert"].median())
        paper_rows[f"likert_iqr_{q}"] = float(q_df["likert"].quantile(0.75) - q_df["likert"].quantile(0.25))
        paper_rows[f"yes_no_majority_{q}"] = majority_yes_over_summaries(q_df["yes_no"])

    paper_summary = pd.DataFrame([paper_rows])
    print("Paper-level summary:")
    print(paper_rows)

    # Write Excel
    folder_output = os.path.dirname(args.input_xlsx)
    base_name = "eval_results_expert_guided.xlsx"
    output_xlsx = os.path.join(folder_output, base_name)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        sampled[["summary_id", "_orig_row", args.ref_col, args.hyp_col]].to_excel(w, sheet_name="sample", index=False)
        raw.to_excel(w, sheet_name="raw_ratings", index=False)
        final_wide.to_excel(w, sheet_name="final_wide_per_summary", index=False)
        paper_summary.to_excel(w, sheet_name="paper_summary", index=False)

    print(f"OK: wrote {output_xlsx}")
    print("Sheets: sample, raw_ratings, final_wide_per_summary, paper_summary")


if __name__ == "__main__":
    main()