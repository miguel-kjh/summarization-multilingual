#!/bin/bash

# model architecture
model_name="Qwen/Qwen2.5-0.5B"
# peft and quantization
peft_type="vera" # lora, dora, vera, loha, lokr
quantization=False
lora_r=256 # for vera 8*8=64
lora_alpha=16
lora_dropout=0.05
lora_target_modules="q_proj,k_proj,v_proj"

# hyperparameters
batch_size=4
learning_rate=1e-4
num_train_epochs=2
weight_decay=0.
context_length=512

# data
dataset_name="data/04-clustering/canario-chunks-sentence-transformers"
wandb=False
# run
python finetuning.py \
    --model_name_or_path $model_name \
    --peft_type $peft_type \
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