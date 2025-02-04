import os
import itertools

# Constants that remain the same for all scripts
CONSTANTS = {
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "batch_size": 1,
    "learning_rate": 1e-4,
    "num_train_epochs": 2,
    "weight_decay": 0.0,
    "context_length": 1024,
    "quantization": False,
    "wandb": False
}

# Lists for varying parameters
MODEL_NAMES = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

PEFT_TYPES = ["lora", "dora", "vera", "loha", "lokr"]

DATASET_NAMES = [
    # normal
    "data/02-processed/spanish",
    "data/02-processed/spanish-reduce",
    # clustering
    "data/04-clustering/spanish-chunks-openai",
    "data/04-clustering/spanish-chunks-sentence-transformers",
    # combined
    "data/03-combined/english-german",
    "data/03-combined/spanish-english",
    "data/03-combined/spanish-french",
    "data/03-combined/spanish-german",
    "data/03-combined/spanish-italian",
    "data/03-combined/spanish-portuguese",
]

# Create an output directory for the scripts
output_dir = "scripts"
os.makedirs(output_dir, exist_ok=True)

# Generate a script for each combination of model, PEFT type, and dataset
for i, (model_name, peft_type, dataset_name) in enumerate(itertools.product(MODEL_NAMES, PEFT_TYPES, DATASET_NAMES)):
    script_filename = os.path.join(output_dir, f"train_{i+1}.sh")

    bash_script = f"""#!/bin/bash

# Model architecture
model_name="{model_name}"

# PEFT and quantization
peft_type="{peft_type}"  # lora, dora, vera, loha, lokr
quantization={CONSTANTS['quantization']}
lora_r={CONSTANTS['lora_r'] if peft_type not in ['vera'] else 256}
lora_alpha={CONSTANTS['lora_alpha']}
lora_dropout={CONSTANTS['lora_dropout']}
lora_target_modules="{CONSTANTS['lora_target_modules']}"

# Hyperparameters
batch_size={CONSTANTS['batch_size']}
learning_rate={CONSTANTS['learning_rate']}
num_train_epochs={CONSTANTS['num_train_epochs']}
weight_decay={CONSTANTS['weight_decay']}
context_length={CONSTANTS['context_length']}

# Data
dataset_name="{dataset_name}"
wandb={CONSTANTS['wandb']}

# Run
python finetuning.py \\
    --model_name_or_path $model_name \\
    --peft_type $peft_type \\
    --lora_target_modules $lora_target_modules \\
    --lora_r $lora_r \\
    --lora_alpha $lora_alpha \\
    --lora_dropout $lora_dropout \\
    --quantization $quantization \\
    --batch_size $batch_size \\
    --lr $learning_rate \\
    --num_train_epochs $num_train_epochs \\
    --weight_decay $weight_decay \\
    --dataset_name $dataset_name \\
    --wandb $wandb \\
    --context $context_length
    """

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")

