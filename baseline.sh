#!/bin/bash

#!/bin/bash

# Lista de idiomas
declare -a languages=("portuguese" "english" "french" "german" "italian")

# Iterar sobre la lista de idiomas
for lang in "${languages[@]}"; do
    dataset="data/02-processed/$lang"
    
    echo "Procesando idioma: $lang"
    
    python3 baseline.py --dataset "$dataset" --method "ghic"
    python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "bert-base-multilingual-cased"
    python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "FacebookAI/xlm-roberta-large"
    # python3 baseline.py --dataset "$dataset" --method "openai"
done
