#!/usr/bin/env bash

# Lista de idiomas (MISMA ORDEN que en los datasets)
declare -a languages=(
    "canario"
    #"spanish"
    #"english"
    #"french"
    #"italian"
    #"portuguese"
    #"german"
)

# Cada posición corresponde al mismo índice en «languages»
declare -a datasets=(
    "data/02-processed/canario"
    #"data/02-processed/spanish"
    #"data/02-processed/english"
    #"data/02-processed/french"
    #"data/02-processed/italian"
    #"data/02-processed/portuguese"
    #"data/02-processed/german"
)

# Lista de modelos
declare -a models=(
    "qwen2.5:7b"
    "llama3.1"
    #"phi4"
    #"qwen3:14b"
    #"qwen3:30b"
)

# ──────────────────────────────────────────────────────────────────────────────
# Recorremos los arrays por ÍNDICE para que lang y dataset siempre coincidan
# ──────────────────────────────────────────────────────────────────────────────
for i in "${!languages[@]}"; do
    lang="${languages[$i]}"
    dataset="${datasets[$i]}"

    echo "Procesando idioma: $lang   |   Dataset: $dataset"

    #python3 baseline.py --dataset "$dataset" --method "textranking" --truncate True --context_window 200000
    #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/textranking/qwen2.5:0.5b" --dataset "$dataset" --method "truncate"

    #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/lsa/qwen2.5:0.5b" --dataset "$dataset" --method "truncate"

    #python3 baseline.py --dataset "$dataset" --method "ghic" --truncate True
    #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/ghic/" --dataset "$dataset" --method "truncate"

    #python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "bert-base-multilingual-cased" --truncate True
    #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/extractive/bert-base-multilingual-cased" --dataset "$dataset" --method "truncate"

    # Extra para inglés
    #if [ "$lang" == "english" ]; then
        #python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "nlpaueb/legal-bert-base-uncased"
        #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/extractive/nlpaueb/legal-bert-base-uncased" --dataset "$dataset" --method "truncate"
    #fi

    # Modelo OpenAI salvo en canario
    #if [ "$lang" != "canario" ]; then
        #python3 baseline.py --dataset "$dataset" --method "openai" --model_name "gpt-4o-mini" --truncate True --context_window 128000
        #python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/openai" --dataset "$dataset" --method "truncate"
    #fi

    # Iterar sobre los modelos Ollama
    for model_name in "${models[@]}"; do
        echo "Procesando modelo: $model_name"
        python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "$model_name"
        python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/ollama/$model_name" --dataset "$dataset" --method "truncate" --recalcule_rouge True --use_openai False
        ollama stop "$model_name"
    done

    echo "-------------------------------------------------------------------"
done
echo "Todos los idiomas procesados."