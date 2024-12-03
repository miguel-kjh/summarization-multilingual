#!/bin/bash

model="models/Llama-3.2-1B-tiny-e2-b2-lr0.0001-wd0.01-c512-r8-a16-d0.05"
wandb=False

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb
