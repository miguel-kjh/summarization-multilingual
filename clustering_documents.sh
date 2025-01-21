#!/bin/bash

dataset_path="data/02-processed/spanish"
model_spacy="es_core_news_sm"
distance_metric="cosine"
wandb=True

# Define las alternativas para method y embedding_model
methods=("chunks" "paragraphs" "sentences")
embedding_models=("sentence-transformers" "openai")

# Bucle para recorrer todas las combinaciones
for method in "${methods[@]}"; do
    for embedding_model in "${embedding_models[@]}"; do
        echo "Running with method=$method and embedding_model=$embedding_model"

        # Ejecuta el script con la combinación actual
        python3 clustering_split.py \
            --dataset_path "$dataset_path" \
            --method "$method" \
            --embedding_model "$embedding_model" \
            --model_spacy "$model_spacy" \
            --distance_metric "$distance_metric" \
            --wandb "$wandb"
    done
done

