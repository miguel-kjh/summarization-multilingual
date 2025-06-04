from unsloth import FastLanguageModel
import os
import warnings

from datetime import datetime
import pandas as pd
import wandb
from lightning import seed_everything
import torch


#WANDB CONFIG
PROJECT_NAME = "eur-lex-sum"

#DATASET
DATASET = "dennlinger/eur-lex-sum"

# Canary
DATASET_PAR_CAN = "miguel-kjh/ParCan-Sum"

# Models
CONTEXT_WINDOWS = {
    "Qwen3": 40000,
    "llama-3.2": 120000,
    "salamandra": 8192
}

# Folders
DATA_FOLDER           = "data"
RAW_DATA_FOLDER       = os.path.join(DATA_FOLDER, "01-raw")
PROCESS_DATA_FOLDER   = os.path.join(DATA_FOLDER, "02-processed")
COMBINED_DATA_FOLDER  = os.path.join(DATA_FOLDER, "03-combined")

# Statistics
FILE_STATS = os.path.join(RAW_DATA_FOLDER, "stats.csv")

# Languages to download
LANGUAGES = ['portuguese', 'spanish', 'english', 'german', 'italian', 'french']
"""
['bulgarian', 'czech', 'dutch', 'estonian', 'french', 'greek', '', 'irish', 'latvian', 'maltese', 'portuguese', 'slovak', 
'spanish', 'croatian', 'danish', 'english', 'finnish', 'german', 'hungarian', 'italian', 'lithuanian', 'polish', 
'romanian', 'slovenian', 'swedish']
"""

# Default filenames
DATASET_FILENAME = "test_summary"
RESULTS_FILENAME = "result_metrics.json"

SEED = 123

INSTRUCTION_TEMPLATE = {
    "english": "Please summarize the following text in a few sentences, highlighting the most important points.",
    "spanish": "Por favor, resuma el siguiente texto en unas pocas frases, destacando los puntos más importantes.",
    "canario": "Por favor, resuma el siguiente texto en unas pocas frases, destacando los puntos más importantes.",
    "german": "Bitte fassen Sie den folgenden Text in ein paar Sätzen zusammen und heben Sie die wichtigsten Punkte hervor.",
    "italian": "Per favore, riassumi il seguente testo in poche frasi, evidenziando i punti più importanti.",
    "portuguese": "Por favor, resuma o texto a seguir em algumas frases, destacando os pontos mais importantes.",
    "french": "Veuillez résumer le texte suivant en quelques phrases, en mettant en évidence les points les plus importants."
}


def generate_prompt(system_prompt: str, document: str,) -> str:
        return f"""### Instruction: {system_prompt}
        
### Input:
{document.strip()}

### Response:""".strip()

def generate_training_prompt(
    system_prompt: str, document: str, summary: str
) -> str:
    
    summary = summary.replace("\n", "")
    return generate_prompt(system_prompt, document) + f"\n{summary.strip()}"

def get_timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")

def setup_environment(args):
    warnings.filterwarnings("ignore")
    if not args.wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = PROJECT_NAME
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=PROJECT_NAME, entity="miguel_kjh", name=args.run_name)
    seed_everything(seed=SEED)
    torch.backends.cudnn.deterministic = True

def generate_names_for_wandb_run(args):
    model_name = args.model_name_or_path.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]
    epochs = args.num_train_epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    neftune_noise_alpha = args.neftune_noise_alpha
    context = args.context
    name_experiment  = f"{model_name}-{dataset_name}-e{epochs}-b{batch_size}-lr{lr}-wd{weight_decay}-c{context}"
    name_experiment += f"-peft-{args.peft_type}-r{args.lora_r}-a{args.lora_alpha}-d{args.lora_dropout}" if args.peft_type else ""
    name_experiment += f"-nna{neftune_noise_alpha}" if neftune_noise_alpha is not None else ""
    name_experiment += "-quant" if args.quantization else ""
    name_experiment += f"-conn-{args.type_connector}" if args.connector else ""
    name_experiment += f"-{get_timestamp()}"
    return name_experiment


def create_model_and_tokenizer(args):
    """if args.quantization:
        print("### Using Quantization ###")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    accelerator = create_accelerator()
    model = accelerator.prepare_model(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model"""

    context_window = next(
        (value for key, value in CONTEXT_WINDOWS.items() if key in args.model_name_or_path),
        None
    )

    # Lanzar excepción si no se encuentra una coincidencia
    if context_window is None:
        raise ValueError(f"Context window not found for model '{args.model_name_or_path}'. Please specify a valid model name.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name_or_path,
        max_seq_length = context_window,
        dtype = None,
        load_in_4bit = args.quantization, # quantization QLoRA 4-bit
    )
    tokenizer.clean_up_tokenization_spaces = False

    return tokenizer, model

def upload_to_wandb(table_name: str, summaries: list):
    df_original = pd.DataFrame(summaries)
    table_name = f"{wandb.run.name}-{table_name}"
    wandb.log(
        {
            table_name: wandb.Table(dataframe=df_original)
        }
    )

def calculate_weighted_mean(metrics: dict) -> float:
    """
    Calculate a normalized mean for metrics with different ranges.

    :param metrics: Dictionary of metrics and their values.
    :return: scaled mean (1-max of scale).
    """
    max_values = {'coherence': 5.0, 'consistency': 5.0, 'fluency': 3.0, 'relevance': 5.0}
    normalized_values = [value / max_values[metric] for metric, value in metrics.items()]
    normalized_mean = sum(normalized_values) / len(normalized_values)
    
    # Optionally scale the mean to a desired range (e.g., 1 to 5)
    scaled_mean = normalized_mean * max(max_values.values())

    return scaled_mean

def wandb_end():
    wandb.finish()

if __name__ == "__main__":
    print("This is a utility module. It should not be run directly.")
    print("Use it as a module in your main script.")
    exit(1)
