import os
import itertools

CONSTANTS = {
    "context_length": 16384,
    "quantization": False, 
    "wandb": True,
}

MODEL_NAMES = [
    "models/others/data_02-processed_french/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_french/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_french/unsloth/Llama-3.2-1B",
    "models/others/data_02-processed_french/unsloth/Llama-3.2-3B-Instruct",
    "models/others/data_02-processed_french/unsloth/Llama-3.2-3B",
    "models/others/data_02-processed_french/unsloth/Llama-3.2-1B-Instruct",
    "models/others/data_02-processed_french/BSC-LT/salamandra-2b-instruct",
    "models/others/data_02-processed_french/BSC-LT/salamandra-2b",
    "models/others/data_02-processed_french/Qwen/Qwen2.5-1.5B-Instruct",
    "models/others/data_02-processed_french/Qwen/Qwen3-1.7B",
    "models/others/data_02-processed_french/Qwen/Qwen2.5-0.5B",
    "models/others/data_02-processed_french/Qwen/Qwen2.5-1.5B",
    "models/others/data_02-processed_french/Qwen/Qwen2.5-0.5B-Instruct",
    "models/others/data_02-processed_french/Qwen/Qwen3-4B-Base",
    "models/others/data_02-processed_french/Qwen/Qwen2.5-3B",
    "models/others/data_02-processed_french/Qwen/Qwen3-1.7B-Base",
    "models/others/data_02-processed_french/Qwen/Qwen3-4B",
    "models/others/data_02-processed_french/Qwen/Qwen2.5-3B-Instruct",
    "models/others/data_02-processed_french/Qwen/Qwen3-0.6B",
    "models/others/data_02-processed_french/Qwen/Qwen3-0.6B-Base",
    "models/others/data_02-processed_english/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_english/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_english/unsloth/Llama-3.2-1B",
    "models/others/data_02-processed_english/unsloth/Llama-3.2-3B-Instruct",
    "models/others/data_02-processed_english/unsloth/Llama-3.2-3B",
    "models/others/data_02-processed_english/unsloth/Llama-3.2-1B-Instruct",
    "models/others/data_02-processed_english/BSC-LT/salamandra-2b-instruct",
    "models/others/data_02-processed_english/BSC-LT/salamandra-2b",
    "models/others/data_02-processed_english/Qwen/Qwen2.5-1.5B-Instruct",
    "models/others/data_02-processed_english/Qwen/Qwen3-1.7B",
    "models/others/data_02-processed_english/Qwen/Qwen2.5-0.5B",
    "models/others/data_02-processed_english/Qwen/Qwen2.5-1.5B",
    "models/others/data_02-processed_english/Qwen/Qwen2.5-0.5B-Instruct",
    "models/others/data_02-processed_english/Qwen/Qwen3-4B-Base",
    "models/others/data_02-processed_english/Qwen/Qwen2.5-3B",
    "models/others/data_02-processed_english/Qwen/Qwen3-1.7B-Base",
    "models/others/data_02-processed_english/Qwen/Qwen3-4B",
    "models/others/data_02-processed_english/Qwen/Qwen2.5-3B-Instruct",
    "models/others/data_02-processed_english/Qwen/Qwen3-0.6B",
    "models/others/data_02-processed_english/Qwen/Qwen3-0.6B-Base",
    "models/others/data_02-processed_portuguese/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_portuguese/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_portuguese/unsloth/Llama-3.2-1B",
    "models/others/data_02-processed_portuguese/unsloth/Llama-3.2-3B-Instruct",
    "models/others/data_02-processed_portuguese/unsloth/Llama-3.2-3B",
    "models/others/data_02-processed_portuguese/unsloth/Llama-3.2-1B-Instruct",
    "models/others/data_02-processed_portuguese/BSC-LT/salamandra-2b-instruct",
    "models/others/data_02-processed_portuguese/BSC-LT/salamandra-2b",
    "models/others/data_02-processed_portuguese/Qwen/Qwen2.5-1.5B-Instruct",
    "models/others/data_02-processed_portuguese/Qwen/Qwen3-1.7B",
    "models/others/data_02-processed_portuguese/Qwen/Qwen2.5-0.5B",
    "models/others/data_02-processed_portuguese/Qwen/Qwen2.5-1.5B",
    "models/others/data_02-processed_portuguese/Qwen/Qwen2.5-0.5B-Instruct",
    "models/others/data_02-processed_portuguese/Qwen/Qwen3-4B-Base",
    "models/others/data_02-processed_portuguese/Qwen/Qwen2.5-3B",
    "models/others/data_02-processed_portuguese/Qwen/Qwen3-1.7B-Base",
    "models/others/data_02-processed_portuguese/Qwen/Qwen3-4B",
    "models/others/data_02-processed_portuguese/Qwen/Qwen2.5-3B-Instruct",
    "models/others/data_02-processed_portuguese/Qwen/Qwen3-0.6B",
    "models/others/data_02-processed_portuguese/Qwen/Qwen3-0.6B-Base",
    "models/others/data_02-processed_italian/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_italian/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_italian/unsloth/Llama-3.2-1B",
    "models/others/data_02-processed_italian/unsloth/Llama-3.2-3B-Instruct",
    "models/others/data_02-processed_italian/unsloth/Llama-3.2-3B",
    "models/others/data_02-processed_italian/unsloth/Llama-3.2-1B-Instruct",
    "models/others/data_02-processed_italian/BSC-LT/salamandra-2b-instruct",
    "models/others/data_02-processed_italian/BSC-LT/salamandra-2b",
    "models/others/data_02-processed_italian/Qwen/Qwen2.5-1.5B-Instruct",
    "models/others/data_02-processed_italian/Qwen/Qwen3-1.7B",
    "models/others/data_02-processed_italian/Qwen/Qwen2.5-0.5B",
    "models/others/data_02-processed_italian/Qwen/Qwen2.5-1.5B",
    "models/others/data_02-processed_italian/Qwen/Qwen2.5-0.5B-Instruct",
    "models/others/data_02-processed_italian/Qwen/Qwen3-4B-Base",
    "models/others/data_02-processed_italian/Qwen/Qwen2.5-3B",
    "models/others/data_02-processed_italian/Qwen/Qwen3-1.7B-Base",
    "models/others/data_02-processed_italian/Qwen/Qwen3-4B",
    "models/others/data_02-processed_italian/Qwen/Qwen2.5-3B-Instruct",
    "models/others/data_02-processed_italian/Qwen/Qwen3-0.6B",
    "models/others/data_02-processed_italian/Qwen/Qwen3-0.6B-Base",
    "models/others/data_02-processed_spanish/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_spanish/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_spanish/unsloth/Llama-3.2-1B",
    "models/others/data_02-processed_spanish/unsloth/Llama-3.2-3B-Instruct",
    "models/others/data_02-processed_spanish/unsloth/Llama-3.2-3B",
    "models/others/data_02-processed_spanish/unsloth/Llama-3.2-1B-Instruct",
    "models/others/data_02-processed_spanish/BSC-LT/salamandra-2b-instruct",
    "models/others/data_02-processed_spanish/BSC-LT/salamandra-2b",
    "models/others/data_02-processed_spanish/Qwen/Qwen2.5-1.5B-Instruct",
    "models/others/data_02-processed_spanish/Qwen/Qwen3-1.7B",
    "models/others/data_02-processed_spanish/Qwen/Qwen2.5-0.5B",
    "models/others/data_02-processed_spanish/Qwen/Qwen2.5-1.5B",
    "models/others/data_02-processed_spanish/Qwen/Qwen2.5-0.5B-Instruct",
    "models/others/data_02-processed_spanish/Qwen/Qwen3-4B-Base",
    "models/others/data_02-processed_spanish/Qwen/Qwen2.5-3B",
    "models/others/data_02-processed_spanish/Qwen/Qwen3-1.7B-Base",
    "models/others/data_02-processed_spanish/Qwen/Qwen3-4B",
    "models/others/data_02-processed_spanish/Qwen/Qwen2.5-3B-Instruct",
    "models/others/data_02-processed_spanish/Qwen/Qwen3-0.6B",
    "models/others/data_02-processed_spanish/Qwen/Qwen3-0.6B-Base",
    "models/others/data_02-processed_canario/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_canario/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_german/unsloth/Llama-3.1-8B-Instruct",
    "models/others/data_02-processed_german/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_german/unsloth/Llama-3.2-1B",
    "models/others/data_02-processed_german/unsloth/Llama-3.2-3B-Instruct",
    "models/others/data_02-processed_german/unsloth/Llama-3.2-3B",
    "models/others/data_02-processed_german/unsloth/Llama-3.2-1B-Instruct",
    "models/others/data_02-processed_german/BSC-LT/salamandra-2b-instruct",
    "models/others/data_02-processed_german/BSC-LT/salamandra-2b",
    "models/others/data_02-processed_german/Qwen/Qwen2.5-1.5B-Instruct",
    "models/others/data_02-processed_german/Qwen/Qwen3-1.7B",
    "models/others/data_02-processed_german/Qwen/Qwen2.5-0.5B",
    "models/others/data_02-processed_german/Qwen/Qwen2.5-1.5B",
    "models/others/data_02-processed_german/Qwen/Qwen2.5-0.5B-Instruct",
    "models/others/data_02-processed_german/Qwen/Qwen3-4B-Base",
    "models/others/data_02-processed_german/Qwen/Qwen2.5-3B",
    "models/others/data_02-processed_german/Qwen/Qwen3-1.7B-Base",
    "models/others/data_02-processed_german/Qwen/Qwen3-4B",
    "models/others/data_02-processed_german/Qwen/Qwen2.5-3B-Instruct",
    "models/others/data_02-processed_german/Qwen/Qwen3-0.6B",
    "models/others/data_02-processed_german/Qwen/Qwen3-0.6B-Base",
]

DATASET_NAMES = [
    ("portuguese", "data/02-processed/portuguese"),
    ("french", "data/02-processed/french") ,
    ("italian", "data/02-processed/italian"),
    ("german", "data/02-processed/german"),
    ("english", "data/02-processed/english"),
    ("spanish", "data/02-processed/spanish"),
    #("canario", "data/02-processed/canario"),
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
    --method "truncate" \\
    --up False
"""

# Create an output directory for the scripts
output_dir = "scripts_2"
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

