#!/bin/bash

model="models/Llama-3.2-1B-tiny-e10-b4-lr0.0001-wd0.0-c512-r8-a16-d0.05-2024-12-12-12-20-13"
wandb=False

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb
