import os
import itertools

CONSTANTS = {
    "context_length": 16384,
    "quantization": False, 
    "wandb": True,
}

MODEL_NAMES = [
    "models/Qwen/Qwen3-4B/spanish/lora/Qwen3-4B-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-08-06-07",
    "models/Qwen/Qwen3-4B/english/lora/Qwen3-4B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-04-08-35",
    "models/Qwen/Qwen3-4B/french/lora/Qwen3-4B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-17-05-56",
    "models/Qwen/Qwen3-4B/german/lora/Qwen3-4B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-01-02-28",
    "models/Qwen/Qwen3-4B/italian/lora/Qwen3-4B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-10-42-20",
    "models/Qwen/Qwen3-4B/portuguese/lora/Qwen3-4B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-21-54-40",
    "models/Qwen/Qwen3-4B-Base/english/lora/Qwen3-4B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-04-05-15",
    "models/Qwen/Qwen3-4B-Base/french/lora/Qwen3-4B-Base-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-21-44-02",
    "models/Qwen/Qwen3-4B-Base/german/lora/Qwen3-4B-Base-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-01-36-50",
    "models/Qwen/Qwen3-4B-Base/italian/lora/Qwen3-4B-Base-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-23-41-59",
    "models/Qwen/Qwen3-4B-Base/portuguese/lora/Qwen3-4B-Base-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-18-14-18",
    "models/Qwen/Qwen3-4B-Base/spanish/lora/Qwen3-4B-Base-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-05-53-08",
    "models/others/data_02-processed_english/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_english/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_french/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_french/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_german/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_german/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_italian/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_italian/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_portuguese/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_portuguese/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_spanish/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
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


def for_evaluation(model_name, dataset_name):
    """
    Generates generation scripts for different combinations of models, PEFT types, and datasets.
    Each script is saved in a specified output directory.
    """
    return f"""
    # Model architecture
    model_name="{model_name}"

    # Data
    dataset_name="{dataset_name}"

    
    python model_evaluate.py \\
    --model $model_name \\
    --verbose True \\
    --dataset $dataset_name \\
    --use_openai True \\
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
    simple_name = model_name.split("/")[2]
    script_filename = os.path.join(output_dir, f"generate_{indx+1}_{simple_name}_{lang}.sh")
    indx += 1
    
    bash_script = for_evaluation(
        model_name=model_name,
        dataset_name=folder_data,
    )

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")

