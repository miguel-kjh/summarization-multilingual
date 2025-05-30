#!/bin/bash

#model="models/baseline/spanish/ollama/qwen2.5:0.5b"
path=models/baseline/
languages=("spanish" "english" "french" "german" "italian" "portuguese")
models=(
    #Qwen/Qwen2.5-3B
    #BSC-LT/salamandra-2b
    #BSC-LT/salamandra-2b-instruct
    meta-llama/Llama-3.2-1B
    #Qwen/Qwen2.5-0.5B
    #Qwen/Qwen2.5-1.5B
    #Qwen/Qwen2.5-3B

)
wandb=False
use_openai=True
method="normal"

# Loop through each model and run the evaluation script
for lang in "${languages[@]}"; do
    echo "Processing language: $lang"
    # Set the dataset path

    for model in "${models[@]}"; do
        echo "Evaluating model: $model"
        # Set the model path
        model_path="models/others/data_02-processed_${lang}/${model}"
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
done
# run
