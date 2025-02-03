#!/bin/bash

model="models/Qwen2.5-0.5B-Instruct-spanish-e1-b1-lr0.0001-wd0.0-c1024-r8-a16-d0.05-quant-2025-01-24-17-52-58"
wandb=False
method="normal"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
