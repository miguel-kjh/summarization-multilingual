#!/bin/bash

#model="models/baseline/spanish/ollama/qwen2.5:0.5b"
models=(
    "models/others/data_02-processed_english/unsloth/Llama-3.2-3B-Instruct"
    "models/others/data_02-processed_english/Qwen/Qwen2.5-3B-Instruct"
    "models/others/data_02-processed_english/Qwen/Qwen3-4B"
    "models/others/data_02-processed_english/BSC-LT/salamandra-2b-instruct"
    "models/Qwen/Qwen2.5-3B-Instruct/english/lora/Qwen2.5-3B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-02-46-25"
    "models/unsloth/Llama-3.2-3B-Instruct/english/lora/Llama-3.2-3B-Instruct-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-23-45-25"
    "models/BSC-LT/salamandra-2b-instruct/english/lora/salamandra-2b-instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-02-41-48"
)
wandb=False
use_openai=True
method="truncate"

# Loop through each model and run the evaluation script
for model in "${models[@]}"; do
    echo "Evaluating model: $model"
    # Set the model path
    model_path="${model}"
    echo "Model path: $model_path"
    
    # Check if the model path exists
    if [ ! -d "$model_path" ]; then
        echo "Model path $model_path does not exist. Skipping..."
        continue
    fi
    
    # Run the evaluation script with the current model
    python3 model_evaluate.py \
        --model_name_or_path "$model_path" \
        --wandb "$wandb" \
        --method "$method" \
        --use_openai "$use_openai" \
        --up False
done
# run
