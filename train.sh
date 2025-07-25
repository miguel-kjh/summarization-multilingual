#!/bin/bash

# Model architecture
model_name="Qwen/Qwen2.5-3B-Instruct"

# PEFT and quantization
peft_type="lora"  # lora, dora, vera, loha, lokr
quantization=False
lora_r=16
lora_dropout=0.0
lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Hyperparameters
batch_size=1
learning_rate=0.0002
num_train_epochs=2
weight_decay=0.0
context_length=8192
eval_steps=1000  # Define eval_steps if needed

# Data
dataset_name="data/02-processed/spanish"
wandb=True

# Run
model_folder=$(python train.py \
    --model_name_or_path $model_name \
    --peft_type $peft_type \
    --lora_target_modules $lora_target_modules \
    --lora_r $lora_r \
    --lora_dropout $lora_dropout \
    --quantization $quantization \
    --batch_size $batch_size \
    --lr $learning_rate \
    --num_train_epochs $num_train_epochs \
    --weight_decay $weight_decay \
    --dataset_name $dataset_name \
    --wandb $wandb \
    --eval_steps $eval_steps \
    --context $context_length | tail -n 1)

python generate.py \
    --model_name_or_path $model_folder \
    --dataset $dataset_name \
    --context_window 16384 \
    --truncate True \
    --using_streamer False \
    --rewrite False \
    --is_adapter True \
    --max_new_tokens 2048 \
    --quantization $quantization \
    
python model_evaluate.py \
    --model $model_folder \
    --verbose True \
    --method "truncate" \
    --use_openai False \
    --up False
