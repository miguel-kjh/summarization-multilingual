#!/bin/bash

#model="models/baseline/spanish/ollama/qwen2.5:0.5b"
models=(
   "models/others/data_02-processed_french/unsloth/Llama-3.2-1B"
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
