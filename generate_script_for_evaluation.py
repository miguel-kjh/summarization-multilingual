import os
import itertools

CONSTANTS = {
    "context_length": 16384,
    "quantization": False, 
    "wandb": True,
}

MODEL_NAMES = [
    "models/BSC-LT/salamandra-2b/canario/lora/salamandra-2b-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-01-50-39",
    "models/others/data_02-processed_canario/BSC-LT/salamandra-2b",
    "models/BSC-LT/salamandra-2b-instruct/canario/lora/salamandra-2b-instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-04-18-33",
    "models/others/data_02-processed_canario/BSC-LT/salamandra-2b-instruct",
    "models/Qwen/Qwen2.5-0.5B/canario/lora/Qwen2.5-0.5B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-03-14-31",
    "models/others/data_02-processed_canario/Qwen/Qwen2.5-0.5B-Instruct",
    "models/others/data_02-processed_canario/Qwen/Qwen2.5-0.5B",
    "models/Qwen/Qwen2.5-0.5B-Instruct/canario/lora/Qwen2.5-0.5B-Instruct-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-14-24-03",
    "models/Qwen/Qwen2.5-1.5B/canario/lora/Qwen2.5-1.5B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-20-18-15",
    "models/others/data_02-processed_canario/Qwen/Qwen2.5-1.5B-Instruct",
    "models/others/data_02-processed_canario/Qwen/Qwen2.5-1.5B",
    "models/Qwen/Qwen2.5-1.5B-Instruct/canario/lora/Qwen2.5-1.5B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-20-23-21",
    "models/Qwen/Qwen2.5-3B/canario/lora/Qwen2.5-3B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-05-34-26",
    "models/others/data_02-processed_canario/Qwen/Qwen2.5-3B-Instruct",
    "models/others/data_02-processed_canario/Qwen/Qwen2.5-3B",
    "models/Qwen/Qwen2.5-3B-Instruct/canario/lora/Qwen2.5-3B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-09-50-52",
    "models/Qwen/Qwen3-0.6B/canario/lora/Qwen3-0.6B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-05-33-26",
    "models/others/data_02-processed_canario/Qwen/Qwen3-0.6B-Base",
    "models/others/data_02-processed_canario/unsloth/Llama-3.2-3B-Instruct",
    "models/unsloth/Llama-3.2-3B/canario/lora/Llama-3.2-3B-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-18-28-10",
    "models/others/data_02-processed_canario/unsloth/Llama-3.2-3B",
    "models/others/data_02-processed_canario/unsloth/Llama-3.1-8B-Instruct",
    "models/unsloth/Llama-3.2-3B-Instruct/canario/lora/Llama-3.2-3B-Instruct-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-01-15-08",
    "models/others/data_02-processed_canario/unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "models/others/data_02-processed_canario/unsloth/Qwen3-8B",
    
]

DATASET_NAMES = [
    #("portuguese", "data/02-processed/portuguese"),
    #("french", "data/02-processed/french") ,
    #("italian", "data/02-processed/italian"),
    #("german", "data/02-processed/german"),
    ("english", "data/02-processed/english"),
    #("spanish", "data/02-processed/spanish"),
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
    --method "truncate" \\
    --use_openai True \\
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

