#!/usr/bin/env bash

# Lista de idiomas (MISMA ORDEN que en los datasets)
declare -a languages=(
    "canario"
    #"english"
    #"spanish"
    #"french"
    #"italian"
    #"portuguese"
    #"german"
)

# Cada posición corresponde al mismo índice en «languages»
declare -a datasets=(
    "data/02-processed/canario"
    #"data/02-processed/english"
    #"data/02-processed/spanish"
    #"data/02-processed/french"
    #"data/02-processed/italian"
    #"data/02-processed/portuguese"
    #"data/02-processed/german"
)

# ──────────────────────────────────────────────────────────────────────────────
# Recorremos los arrays por ÍNDICE para que lang y dataset siempre coincidan
# ──────────────────────────────────────────────────────────────────────────────
for i in "${!languages[@]}"; do
    lang="${languages[$i]}"
    dataset="${datasets[$i]}"

    echo "Procesando idioma: $lang   |   Dataset: $dataset"

    python3 baseline.py --dataset "$dataset" --method "lsa" --truncate True
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/lsa/qwen2.5:0.5b" --dataset "$dataset" --method "truncate" --use_openai True

    python3 baseline.py --dataset "$dataset" --method "textranking" --truncate True --context_window 200000
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/textranking/qwen2.5:0.5b" --dataset "$dataset" --method "truncate" --use_openai True

    python3 baseline.py --dataset "$dataset" --method "ghic" --truncate True
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/ghic/" --dataset "$dataset" --method "truncate" --use_openai True

    python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "bert-base-multilingual-cased" --truncate True
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/extractive/bert-base-multilingual-cased" --dataset "$dataset" --method "truncate" --use_openai True

    # Extra para inglés
    if [ "$lang" == "english" ]; then
        python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "nlpaueb/legal-bert-base-uncased"
        python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/extractive/nlpaueb/legal-bert-base-uncased" --dataset "$dataset" --method "truncate" --use_openai True
    fi

    # Modelo OpenAI salvo en canario
    python3 baseline.py --dataset "$dataset" --method "openai" --model_name "gpt-4o-mini" --truncate True --context_window 128000
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/openai" --dataset "$dataset" --method "truncate" --use_openai True --use_openai True
    echo "-------------------------------------------------------------------"
done
echo "Todos los idiomas procesados."