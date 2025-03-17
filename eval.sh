#!/bin/bash

model="models/baseline/portuguese/ghic"
wandb=False
use_openai=False
method="normal"

# run
python model_evaluate.py \
    --model_name_or_path $model \
    --wandb $wandb \
    --method $method \
    --use_openai $use_openai \
    --up False \
