#!/bin/bash

dataset="data/02-processed/spanish"

python3 baseline.py  --dataset $dataset --method "ghic"
python3 baseline.py  --dataset $dataset --method "extractive" --model_name "bert-base-multilingual-cased"
python3 baseline.py  --dataset $dataset --method "extractive" --model_name "FacebookAI/xlm-roberta-large"
python3 baseline.py  --dataset $dataset --method "openai"
