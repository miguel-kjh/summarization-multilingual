#!/bin/bash

model="models/Qwen2.5-0.5B-Instruct-spanish-paragraphs-sentence-transformers-e2-b1-lr0.0001-wd0.0-c1024-r8-a16-d0.05-quant-2025-01-25-15-31-30"
wandb=False
method="clustering"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
