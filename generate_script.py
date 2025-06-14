import os
import itertools

FOR_TRAINING = False  # Set to True for training scripts, False for generation scripts

# Constants that remain the same for all scripts
CONSTANTS = {
    "lora_r": 16,
    "lora_dropout": 0.0,
    "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "batch_size": 2,
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "weight_decay": 0.0,
    "context_length": 18000, #8192,
    "quantization": False, 
    "wandb": True,
}

# Lists for varying parameters
MODEL_NAMES = [
    # qwen 2.5
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # qwen 3
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    # llama 3.2
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
]

PEFT_TYPES = ["lora"]

DATASET_NAMES = [
    "data/02-processed/portuguese",
    "data/02-processed/french",
    "data/02-processed/italian",
    "data/02-processed/german",
    "data/02-processed/english",
    "data/02-processed/spanish",
    #"data/02-processed/canario",
]

# scripts funct

def for_training(model_name, peft_type, dataset_name, max_new_tokens, eval_steps):
    """
    Generates training scripts for different combinations of models, PEFT types, and datasets.
    Each script is saved in a specified output directory.
    """
    return f"""#!/bin/bash

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
eval_steps={eval_steps}  # Define eval_steps if needed

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
    --eval_steps $eval_steps \\
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

def for_generation(model_name, dataset_name, max_new_tokens):
    """
    Generates generation scripts for different combinations of models, PEFT types, and datasets.
    Each script is saved in a specified output directory.
    """
    return f"""
    # Model architecture
    model_name="{model_name}"

    # quantization
    quantization={CONSTANTS['quantization']}

    # Data
    dataset_name="{dataset_name}"
    context_length={CONSTANTS['context_length']}
    
    model_folder=$(python generate.py \\
    --model_name_or_path $model_name \\
    --dataset $dataset_name \\
    --context_window $context_length \\
    --using_streamer False \\
    --using_clustering False \\
    --rewrite True \\
    --max_new_tokens {max_new_tokens} \\
    --quantization $quantization | tail -n 1)

    
    python model_evaluate.py \\
    --model $model_folder \\
    --verbose True \\
    --method "normal" \\
    --up False
"""

# Create an output directory for the scripts
output_dir = "scripts"
os.makedirs(output_dir, exist_ok=True)

# Generate a script for each combination of model, PEFT type, and dataset
for i, (model_name, peft_type, dataset_name) in enumerate(itertools.product(MODEL_NAMES, PEFT_TYPES, DATASET_NAMES)):
    max_new_tokens = 1345 if "canario" in dataset_name else 2048
    eval_steps = 100 if "llama" in model_name else 1000
    simple_name = model_name.split("/")[-1]
    script_filename = os.path.join(output_dir, f"train_{i+1}_{simple_name}_{peft_type}.sh")

    if FOR_TRAINING:
        bash_script = for_training(
            model_name=model_name,
            peft_type=peft_type,
            dataset_name=dataset_name,
            max_new_tokens=max_new_tokens,
            eval_steps=eval_steps
        )
    else:
        bash_script = for_generation(
            model_name=model_name,
            dataset_name=dataset_name,
            max_new_tokens=max_new_tokens
        )

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")

