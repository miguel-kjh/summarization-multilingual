#!/bin/bash

dataset="data/02-processed/spanish"

python3 baseline.py  --dataset $dataset --method "ghic"
python3 model_evaluate.py --model_name_or_path models/baseline/spanish_ghic --wandb True
python3 baseline.py  --dataset $dataset --method "extractive" --model_name "bert-base-multilingual-cased"
python3 model_evaluate.py --model_name_or_path models/baseline/spanish_extractive_bert-base-multilingual-cased --wandb True
python3 baseline.py  --dataset $dataset --method "extractive" --model_name "FacebookAI/xlm-roberta-large"
python3 model_evaluate.py --model_name_or_path models/baseline/spanish_extractive_FacebookAI/xlm-roberta-large --wandb True
python3 baseline.py  --dataset $dataset --method "openai"
python3 model_evaluate.py --model_name_or_path models/baseline/spanish_openai --wandb True