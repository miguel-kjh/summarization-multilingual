#!/bin/bash

model="models/Qwen2.5-0.5B-Instruct-spanish-chunks-openai-e2-b1-lr0.0001-wd0.0-c1024-peft-lora-r8-a16-d0.05-2025-01-27-22-34-11"
wandb=False
method="clustering"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
