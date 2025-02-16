#!/bin/bash

#model="models/others/Qwen2.5-0.5B-Instruct-spanish-paragraphs-sentence-transformers-e2-b1-lr0.0001-wd0.0-c1024-r8-a16-d0.05-quant-2025-01-25-15-31-30"
model="models/baseline/spanish_openai"
wandb=True
method="normal"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
