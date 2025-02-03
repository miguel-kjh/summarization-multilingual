#!/bin/bash

dataset_path=("data/02-processed/french" "data/02-processed/german" "data/02-processed/italian" "data/02-processed/portuguese" "data/02-processed/english")
model_spacy="es_core_news_sm"
distance_metric="cosine"
wandb=True

# Define las alternativas para method y embedding_model
methods="paragraphs"
embedding_models="sentence-transformers"

# Bucle para recorrer todas las combinaciones
for data in "${dataset_path[@]}"; do
    echo "Running with dataset: $data"

    # Ejecuta el script con la combinación actual
    python3 clustering_split.py \
        --dataset_path "$data" \
        --method "$methods" \
        --embedding_model "$embedding_models" \
        --model_spacy "$model_spacy" \
        --distance_metric "$distance_metric" \
        --wandb "$wandb"
done

