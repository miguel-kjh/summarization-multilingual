#!/bin/bash

#!/bin/bash

# Lista de idiomas
declare -a languages=(
    #canario"
    "spanish"
    "english"
    "french"
    "german"
    "italian"
    "portuguese"
    "german"
)
# Lista de modelos
declare -a models=(
    "qwen3:30b"
)

# Iterar sobre la lista de idiomas
for lang in "${languages[@]}"; do
    dataset="data/02-processed/$lang"
    
    echo "Procesando idioma: $lang"
    
    python3 baseline.py --dataset "$dataset" --method "textranking" --truncate True --context_window 200000
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/textranking/qwen2.5:0.5b" --method "truncate" --use_openai False
    python3 baseline.py --dataset "$dataset" --method "ghic" --truncate True
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/ghic/" --method "truncate" --use_openai False
    python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "bert-base-multilingual-cased" --truncate True
    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/extractive/bert-base-multilingual-cased" --method "truncate" --use_openai False
    if [ "$lang" == "english" ]; then
        python3 baseline.py --dataset "$dataset" --method "extractive" --model_name "nlpaueb/legal-bert-base-uncased" --truncate True
        python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/extractive/nlpaueb/legal-bert-base-uncased" --method "truncate" --use_openai False
    fi
    # si no es canario ejecuta el modelo de openai
    if [ "$lang" != "canario" ]; then
        python3 baseline.py --dataset "$dataset" --method "openai" --model_name "gpt-4o-mini" --truncate True --context_window 128000
        python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/openai" --method "truncate" --use_openai False
    fi

    #for model_name in "${models[@]}"; do
    #    echo "Procesando modelo: $model_name"
    #    python3 baseline.py --dataset "$dataset" --method "ollama" --model_name "$model_name"
    #    python3 model_evaluate.py --model_name_or_path "models/baseline/$lang/ollama/$model_name"
    #    ollama stop "$model_name"
    #done

done
