#!/bin/bash

model="models/Qwen/Qwen2.5-0.5B/spanish-chunks-sentence-transformers/lora/Qwen2.5-0.5B-spanish-chunks-sentence-transformers-e2-b1-lr0.0001-wd0.0-c1024-peft-lora-r8-a16-d0.05-2025-02-13-07-33-56"
wandb=False
use_openai=True
method="clustering"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
    --use_openai $use_openai \
    --up $up \
