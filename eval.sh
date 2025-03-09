#!/bin/bash

model="models/meta-llama/Llama-3.2-1B-Instruct/canario-chunks-sentence-transformers/lora/Llama-3.2-1B-Instruct-canario-chunks-sentence-transformers-e1-b4-lr0.0001-wd0.0-c512-peft-lora-r8-a16-d0.05-2025-03-07-04-31-52"
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
