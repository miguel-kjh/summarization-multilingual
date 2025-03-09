#!/bin/bash

model="models/Qwen/Qwen2.5-1.5B-Instruct/canario-chunks-sentence-transformers/lora/Qwen2.5-1.5B-Instruct-canario-chunks-sentence-transformers-e1-b4-lr0.0001-wd0.0-c512-peft-lora-r8-a16-d0.05-2025-03-06-22-23-36"
wandb=False
use_openai=True
method="clustering"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
    --use_openai $use_openai \
    --up True \
