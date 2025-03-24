#!/bin/bash

dataset_path=(
    "data/02-processed/french" 
    "data/02-processed/german" 
    "data/02-processed/italian" 
    "data/02-processed/portuguese" 
    "data/02-processed/english"
    "data/03-combined/english-german"
    "data/03-combined/spanish-english"
    "data/03-combined/spanish-french"
    "data/03-combined/spanish-german"
    "data/03-combined/spanish-italian"
    "data/03-combined/spanish-portuguese"
    "data/02-processed/canario"
)
model_spacy="es_core_news_sm"
distance_metric="cosine"
wandb=True

# Define las alternativas para method y embedding_model
methods="sentences"
embedding_models=("sentence-transformers")

# Bucle para recorrer todas las combinaciones
echo "Running clustering with sentences"
for data in "${dataset_path[@]}"; do
    echo "Running with dataset: $data"
    for embedding_model in "${embedding_models[@]}"; do
        echo "Running with embedding model: $embedding_model"
        # Ejecuta el script con la combinación actual
        python3 clustering_split.py \
            --dataset_path "$data" \
            --method "$methods" \
            --embedding_model "$embedding_model" \
            --model_spacy "$model_spacy" \
            --distance_metric "$distance_metric" \
            --wandb "$wandb" \
            --percentage_of_data 0.3
    done
done

