#!/bin/bash

# model architecture
model_name="BSC-LT/salamandra-7b"
# peft and quantization
lora=True
quantization=True
lora_r=4
lora_alpha=8
lora_dropout=0.05
lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# hyperparameters
batch_size=2
learning_rate=1e-4
num_train_epochs=10
weight_decay=0.
context_length=256

# data
dataset_name="data/02-processed/spanish"
wandb=True
# run
python finetuning.py \
    --model_name_or_path $model_name \
    --lora $lora \
    --lora_target_modules $lora_target_modules \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --quantization $quantization \
    --batch_size $batch_size \
    --lr $learning_rate \
    --num_train_epochs $num_train_epochs \
    --weight_decay $weight_decay \
    --dataset_name $dataset_name \
    --wandb $wandb \
    --context $context_length \



