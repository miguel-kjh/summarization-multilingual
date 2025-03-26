#!/bin/bash

model="models/BSC-LT/salamandra-2b-instruct/canario-chunks-sentence-transformers/lora/salamandra-2b-instruct-canario-chunks-sentence-transformers-e2-b2-lr0.0001-wd0.0-c256-peft-lora-r8-a16-d0.05-2025-03-18-23-47-34"
wandb=False
use_openai=False
method="clustering"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
    --use_openai $use_openai \
    --up False \
