import os
import itertools

# Constants that remain the same for all scripts
CONSTANTS = {
    "lora_r": 16,
    "lora_dropout": 0.0,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "batch_size": 1,
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "weight_decay": 0.0,
    "context_length": 8192,
    "quantization": False, 
    "wandb": True,
}

# Lists for varying parameters
MODEL_NAMES = [
    "BSC-LT/salamandra-2b-instruct",
    "BSC-LT/salamandra-2b",
]

PEFT_TYPES = ["lora"]

DATASET_NAMES = [
    "data/02-processed/portuguese",
    "data/02-processed/french",
    "data/02-processed/italian",
    "data/02-processed/german",
    "data/02-processed/english",
    "data/02-processed/spanish",
    "data/02-processed/canario",
]

# Create an output directory for the scripts
output_dir = "scripts"
os.makedirs(output_dir, exist_ok=True)

# Generate a script for each combination of model, PEFT type, and dataset
for i, (model_name, peft_type, dataset_name) in enumerate(itertools.product(MODEL_NAMES, PEFT_TYPES, DATASET_NAMES)):
    max_new_tokens = 1345 if "canario" in dataset_name else 2048
    simple_name = model_name.split("/")[-1]
    script_filename = os.path.join(output_dir, f"train_{i+1}_{simple_name}_{peft_type}.sh")

    bash_script = f"""#!/bin/bash

# Model architecture
model_name="{model_name}"

# PEFT and quantization
peft_type="{peft_type}"  # lora, dora, vera, loha, lokr
quantization={CONSTANTS['quantization']}
lora_r={CONSTANTS['lora_r'] if peft_type not in ['vera'] else 256}
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
model_folder=$(python train.py \\
    --model_name_or_path $model_name \\
    --peft_type $peft_type \\
    --lora_target_modules $lora_target_modules \\
    --lora_r $lora_r \\
    --lora_dropout $lora_dropout \\
    --quantization $quantization \\
    --batch_size $batch_size \\
    --lr $learning_rate \\
    --num_train_epochs $num_train_epochs \\
    --weight_decay $weight_decay \\
    --dataset_name $dataset_name \\
    --wandb $wandb \\
    --context $context_length | tail -n 1)

python generate.py \\
    --model_name_or_path $model_folder \\
    --dataset $dataset_name \\
    --context_window $context_length \\
    --using_streamer False \\
    --using_clustering False \\
    --rewrite False \\
    --max_new_tokens {max_new_tokens} \\
    --quantization $quantization \\
    
python model_evaluate.py \\
    --model $model_folder \\
    --verbose True \\
    --method "normal" \\
    --up False
    
"""

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")

