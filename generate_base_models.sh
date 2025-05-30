models=(
    "meta-llama/Llama-3.2-1B"
)

languages=(
    "spanish"
    "english"
    "french"
    "german"
    "italian"
    "portuguese"
)
# Loop through each language and model

for lang in "${languages[@]}"; do
    echo "Processing language: $lang"
    dataset="data/02-processed/$lang"
    
    for model_name in "${models[@]}"; do
        echo "Processing model: $model_name"
        python3 generate.py --dataset "$dataset" --model_name "$model_name"
    done
done