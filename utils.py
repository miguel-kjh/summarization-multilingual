import re
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
    "spanish": "Redacta un resumen institucional en español del siguiente documento. Mantén un lenguaje objetivo, enfocado en los hechos y acuerdos:",
    "canario": "Redacta un resumen institucional en español del siguiente documento. Mantén un lenguaje objetivo, enfocado en los hechos y acuerdos:",
    "english": "Write an institutional summary in English of the following document. Keep the language objective, focusing on facts and agreements:",
    "german": "Schreiben Sie eine institutionelle Zusammenfassung des folgenden Dokuments auf Deutsch. Halten Sie die Sprache objektiv und konzentrieren Sie sich auf Fakten und Vereinbarungen:",
    "french": "Rédigez un résumé institutionnel en français du document suivant. Gardez un langage objectif, axé sur les faits et les accords :",
    "italian": "Scrivi un riassunto istituzionale in italiano del seguente documento. Mantieni un linguaggio obiettivo, concentrandoti su fatti e accordi:",
    "portuguese": "Escreva um resumo institucional em português do seguinte documento. Mantenha uma linguagem objetiva, focada em fatos e acordos:",
}

SYSTEM_PROMPT = {
    "canario": "Eres un modelo entrenado para generar resúmenes institucionales de actas parlamentarias. Los resúmenes deben estar redactados en lenguaje formal-administrativo, sin juicios de valor, y seguir una estructura clara.",
    "spanish": "Eres un modelo entrenado para generar resúmenes institucionales de actas parlamentarias. Los resúmenes deben estar redactados en lenguaje formal-administrativo, sin juicios de valor, y seguir una estructura clara.",
    "english": "You are a model trained to generate institutional summaries of parliamentary minutes. The summaries should be written in formal-administrative language, without value judgments, and follow a clear structure.",
    "german": "Sie sind ein Modell, das darauf trainiert ist, institutionelle Zusammenfassungen von Parlamentsprotokollen zu erstellen. Die Zusammenfassungen sollten in formeller Verwaltungssprache verfasst sein, ohne Werturteile, und einer klaren Struktur folgen.",
    "french": "Vous êtes un modèle entraîné pour générer des résumés institutionnels des procès-verbaux parlementaires. Les résumés doivent être rédigés dans un langage formel-administratif, sans jugements de valeur, et suivre une structure claire.",
    "italian": "Sei un modello addestrato per generare riassunti istituzionali dei verbali parlamentari. I riassunti devono essere redatti in linguaggio formale-amministrativo, senza giudizi di valore, e seguire una struttura chiara.",
    "portuguese": "Você é um modelo treinado para gerar resumos institucionais de atas parlamentares. Os resumos devem ser escritos em linguagem formal-administrativa, sem julgamentos de valor, e seguir uma estrutura clara.",
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
    context = args.context
    name_experiment  = f"{model_name}-{dataset_name}-e{epochs}-b{batch_size}-lr{lr}-wd{weight_decay}-c{context}"
    name_experiment += f"-peft-{args.peft_type}-r{args.lora_r}-a{args.lora_r*2}-d{args.lora_dropout}" if args.peft_type else ""
    name_experiment += "-quant" if args.quantization else ""
    name_experiment += f"-{get_timestamp()}"
    return name_experiment

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


def extract_clean_assistant_response(full_text: str) -> str:
    # Buscar el último bloque <|assistant|>
    assistant_start = full_text.rfind("assistant")
    if assistant_start == -1:
        assistant_content = full_text
    else:
        assistant_content = full_text[assistant_start + len("assistant"):]

    # Eliminar los bloques <think>...</think> si existen
    assistant_content = re.sub(r"<think>.*?</think>", "", assistant_content, flags=re.DOTALL)

    # Eliminar espacios extra al principio y al final
    return assistant_content.strip()

def apply_chat_template(instruction: str, system_prompt: str, sample: dict, tokenizer) -> str:
    """
    Apply a chat template to the sample.
    """
    # Define the chat template
    empty_prompt = f"{instruction}\n ##Documento {{document}}\n ##Resumen:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": empty_prompt.format(document=sample["document"])},
    ]
    
    # Format the chat template with the sample text
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def count_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params
    return trainable_params, total_params, percentage
