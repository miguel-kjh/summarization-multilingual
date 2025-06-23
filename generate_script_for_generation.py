import os
import itertools

CONSTANTS = {
    "context_length": 10000,
    "quantization": False, 
    "wandb": True,
}

MODEL_NAMES = [
    "models/unsloth/Llama-3.2-1B/english/lora/Llama-3.2-1B-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-11-20-00",
    "models/unsloth/Llama-3.2-1B/french/lora/Llama-3.2-1B-french-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-10-25-07",
    "models/unsloth/Llama-3.2-1B/canario/lora/Llama-3.2-1B-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-12-22-29",
    "models/unsloth/Llama-3.2-1B/italian/lora/Llama-3.2-1B-italian-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-10-46-09",
    "models/unsloth/Llama-3.2-1B/german/lora/Llama-3.2-1B-german-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-23-10-37-25",
    "models/unsloth/Llama-3.2-1B/spanish/lora/Llama-3.2-1B-spanish-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-11-59-14",
    "models/unsloth/Llama-3.2-1B/portuguese/lora/Llama-3.2-1B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-09-30-28",
    "models/unsloth/Llama-3.2-3B-Instruct/english/lora/Llama-3.2-3B-Instruct-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-23-45-25",
    "models/unsloth/Llama-3.2-3B-Instruct/french/lora/Llama-3.2-3B-Instruct-french-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-21-40-38",
    "models/unsloth/Llama-3.2-3B-Instruct/canario/lora/Llama-3.2-3B-Instruct-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-01-15-08",
    "models/unsloth/Llama-3.2-3B-Instruct/italian/lora/Llama-3.2-3B-Instruct-italian-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-22-23-51",
    "models/unsloth/Llama-3.2-3B-Instruct/german/lora/Llama-3.2-3B-Instruct-german-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-23-03-18",
    "models/unsloth/Llama-3.2-3B-Instruct/spanish/lora/Llama-3.2-3B-Instruct-spanish-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-00-33-26",
    "models/unsloth/Llama-3.2-3B-Instruct/portuguese/lora/Llama-3.2-3B-Instruct-portuguese-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-19-42-53",
    "models/unsloth/Llama-3.2-3B/english/lora/Llama-3.2-3B-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-16-48-08",
    "models/unsloth/Llama-3.2-3B/french/lora/Llama-3.2-3B-french-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-14-28-40",
    "models/unsloth/Llama-3.2-3B/canario/lora/Llama-3.2-3B-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-18-28-10",
    "models/unsloth/Llama-3.2-3B/italian/lora/Llama-3.2-3B-italian-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-15-23-13",
    "models/unsloth/Llama-3.2-3B/german/lora/Llama-3.2-3B-german-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-16-05-00",
    "models/unsloth/Llama-3.2-3B/spanish/lora/Llama-3.2-3B-spanish-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-17-43-51",
    "models/unsloth/Llama-3.2-3B/portuguese/lora/Llama-3.2-3B-portuguese-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-13-33-17",
    "models/unsloth/Llama-3.2-1B-Instruct/english/lora/Llama-3.2-1B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-16-28-29",
    "models/unsloth/Llama-3.2-1B-Instruct/french/lora/Llama-3.2-1B-Instruct-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-14-50-39",
    "models/unsloth/Llama-3.2-1B-Instruct/canario/lora/Llama-3.2-1B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-17-37-13",
    "models/unsloth/Llama-3.2-1B-Instruct/italian/lora/Llama-3.2-1B-Instruct-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-15-23-56",
    "models/unsloth/Llama-3.2-1B-Instruct/german/lora/Llama-3.2-1B-Instruct-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-15-55-19",
    "models/unsloth/Llama-3.2-1B-Instruct/spanish/lora/Llama-3.2-1B-Instruct-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-17-04-30",
    "models/unsloth/Llama-3.2-1B-Instruct/portuguese/lora/Llama-3.2-1B-Instruct-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-14-19-54",
]

DATASET_NAMES = [
    ("portuguese", "data/02-processed/portuguese"),
    ("french", "data/02-processed/french") ,
    ("italian", "data/02-processed/italian"),
    ("german", "data/02-processed/german"),
    ("english", "data/02-processed/english"),
    ("spanish", "data/02-processed/spanish"),
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

