#!/bin/bash

#model="models/baseline/spanish/ollama/qwen2.5:0.5b"
path=models/baseline/
languages=(
    "canario"
)
models=(
    #Qwen/Qwen2.5-3B
    #BSC-LT/salamandra-2b
    #BSC-LT/salamandra-2b-instruct
    models/meta-llama/Llama-3.2-3B-Instruct/canario-chunks-sentence-transformers/lora/Llama-3.2-3B-Instruct-canario-chunks-sentence-transformers-e2-b2-lr0.0001-wd0.0-c256-peft-lora-r8-a16-d0.05-2025-03-21-08-45-16
    #Qwen/Qwen2.5-0.5B
    #Qwen/Qwen2.5-1.5B
    #Qwen/Qwen2.5-3B

)
wandb=False
use_openai=True
method="clustering"

# Loop through each model and run the evaluation script
for lang in "${languages[@]}"; do
    echo "Processing language: $lang"
    # Set the dataset path

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
            --up True
    done
done
# run
