#!/bin/bash

model="models/Llama-3.2-1B-spanish_sentences_clustering-e1-b2-lr0.0001-wd0.0-c1024-r8-a16-d0.05-quant-2025-01-10-17-39-44"
wandb=True
method="clf"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
