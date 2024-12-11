#!/bin/bash

# model architecture
model_name="EleutherAI/pythia-70m"
# peft and quantization
lora=False
quantization=False
lora_r=8
lora_alpha=16
lora_dropout=0.05
lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# hyperparameters
batch_size=4
learning_rate=1e-4
num_train_epochs=10
weight_decay=0.
context_length=512

# connector
connector="models/connectors/connector_pythia-410m-pythia-70m-tiny-e10-b4-c512-id512-2024-12-11-11-06-53"
type_connector="pythia-410m"

# data
dataset_name="data/03-combined/tiny"
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
    --connector $connector \
    --type_connector $type_connector



