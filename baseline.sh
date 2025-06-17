#!/bin/bash

#!/bin/bash

# Lista de idiomas
declare -a languages=("spanish" "english" "french" "german" "italian" "portuguese")
# Lista de modelos
declare -a models=(
    "qwen2.5:7b" 
    "qwen2.5:14b"
)

# Iterar sobre la lista de idiomas
for lang in "${languages[@]}"; do
    dataset="data/02-processed/$lang"
    
    echo "Procesando idioma: $lang"
    
    python3 baseline.py --dataset "$dataset" --method "openai" --model_name "gpt-4o-mini"
    #python3 baseline.py --dataset "$dataset" --method "ghic"
    #python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "bert-base-multilingual-cased"
    #python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "FacebookAI/xlm-roberta-large"
    #python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "phi4"
    #ollama stop phi4
    #python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "llama3.1"
    #ollama stop llama3.1

    #for model_name in "${models[@]}"; do
    #    echo "Procesando modelo: $model_name"
    #    python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "$model_name"
    #    ollama stop "$model_name"
    #done

done
