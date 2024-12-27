#!/bin/bash

model="models/Llama-3.2-3B-spanish-e10-b2-lr0.0001-wd0.0-c512-r8-a16-d0.05-quant-2024-12-14-20-10-15"
wandb=True

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb
