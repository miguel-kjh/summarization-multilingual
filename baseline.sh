#!/bin/bash

#!/bin/bash

# Lista de idiomas
declare -a languages=("canario")
# Lista de modelos
declare -a models=(
    "qwen2.5:7b" 
    "llama3.1"
    "phi4"
    "qwen2.5:14b"
    "qwen3:14b"

)

# Iterar sobre la lista de idiomas
for lang in "${languages[@]}"; do
    dataset="data/02-processed/$lang"
    
    echo "Procesando idioma: $lang"
    
    #python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "nlpaueb/legal-bert-base-uncased"
    #python3 baseline.py --dataset "$dataset" --method "textranking"
    #python3 model_evaluate.py --model_name_or_path "models/baseline/canario/textranking/qwen2.5:0.5b"
    #python3 baseline.py --dataset "$dataset" --method "openai" --model_name "gpt-4o-mini"
    #python3 baseline.py --dataset "$dataset" --method "ghic"
    #python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "bert-base-multilingual-cased"
    #python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "FacebookAI/xlm-roberta-large"
    #python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "phi4"
    #ollama stop phi4
    #python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "llama3.1"
    #ollama stop llama3.1

    for model_name in "${models[@]}"; do
        echo "Procesando modelo: $model_name"
        python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "$model_name"
        python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/ollama/$model_name"
        ollama stop "$model_name"
    done

done
