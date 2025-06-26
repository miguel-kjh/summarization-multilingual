import os
import itertools

CONSTANTS = {
    "context_length": 16384,
    "quantization": False, 
    "wandb": True,
}

MODEL_NAMES = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Base",
]

DATASET_NAMES = [
    ("portuguese", "data/02-processed/portuguese"),
    ("french", "data/02-processed/french") ,
    ("italian", "data/02-processed/italian"),
    ("german", "data/02-processed/german"),
    ("english", "data/02-processed/english"),
    ("spanish", "data/02-processed/spanish"),
    ("canario", "data/02-processed/canario"),
]


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
    --is_adapter True \\
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
indx = 0
for (model_name, dataset_name) in itertools.product(MODEL_NAMES, DATASET_NAMES):
    lang = dataset_name[0]
    if lang not in model_name:
        continue
    folder_data = dataset_name[1]
    max_new_tokens = 1345 if "canario" in dataset_name else 2048
    simple_name = model_name.split("/")[2]
    script_filename = os.path.join(output_dir, f"generate_{indx+1}_{simple_name}_{lang}.sh")
    indx += 1
    
    bash_script = for_generation(
        model_name=model_name,
        dataset_name=folder_data,
        max_new_tokens=max_new_tokens
    )

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")

