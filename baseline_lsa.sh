#!/usr/bin/env bash

# Lista de idiomas (MISMA ORDEN que en los datasets)
declare -a languages=(
    "spanish"
    "english"
    "french"
    "italian"
    "portuguese"
    "german"
    "canario"
)

# Cada posición corresponde al mismo índice en «languages»
declare -a datasets=(
    "data/02-processed/spanish"
    "data/02-processed/english"
    "data/02-processed/french"
    "data/02-processed/italian"
    "data/02-processed/portuguese"
    "data/02-processed/german"
    "data/02-processed/canario"
)

# ──────────────────────────────────────────────────────────────────────────────
# Recorremos los arrays por ÍNDICE para que lang y dataset siempre coincidan
# ──────────────────────────────────────────────────────────────────────────────
for i in "${!languages[@]}"; do
    lang="${languages[$i]}"
    dataset="${datasets[$i]}"

    echo "Procesando idioma: $lang   |   Dataset: $dataset"

    if [ "$lang" != "canario" ]; then
        python3 baseline.py --dataset "$dataset" --method "lsa" --truncate True
        python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/lsa/qwen2.5:0.5b" --dataset "$dataset" --method "truncate" --use_openai False --recalcule_rouge True
        
        #python3 baseline.py --dataset "$dataset" --method "kl" --truncate True
        #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/kl/qwen2.5:0.5b" --dataset "$dataset" --method "truncate" --use_openai False
    fi

    #python3 baseline.py --dataset "$dataset" --method "kl"
    #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/kl/qwen2.5:0.5b" --dataset "$dataset"


    echo "-------------------------------------------------------------------"
done
echo "Todos los idiomas procesados."