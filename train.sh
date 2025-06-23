#!/bin/bash

# Model architecture
model_name="unsloth/Llama-3.2-1B"  # Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct

# PEFT and quantization
peft_type="lora"  # lora, dora, vera, loha, lokr
quantization=False
lora_r=16
lora_dropout=0.0
lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Hyperparameters
batch_size=1
learning_rate=0.0002
num_train_epochs=1
weight_decay=0.0
context_length=8192
eval_steps=1000  # Define eval_steps if needed

# Data
dataset_name="data/02-processed/german"
wandb=False

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
    --is_adapter True \
    --context_window 10000 \
    --using_streamer False \
    --using_clustering False \
    --rewrite False \
    --max_new_tokens 2048 \
    --quantization $quantization \
    
python model_evaluate.py \
    --model $model_folder \
    --verbose True \
    --method "normal" \
    --up False
    
