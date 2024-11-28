import os
import warnings

import wandb
from lightning import seed_everything
import torch

#WANDB CONFIG
PROJECT_NAME = "multilingual-summarization"

# Folders
DATA_FOLDER           = "data"
RAW_DATA_FOLDER       = os.path.join(DATA_FOLDER, "01-raw")
PROCESS_DATA_FOLDER   = os.path.join(DATA_FOLDER, "02-processed")
COMBINED_DATA_FOLDER  = os.path.join(DATA_FOLDER, "03-combined")

# Statistics
FILE_STATS = os.path.join(RAW_DATA_FOLDER, "stats.csv")

# Languages to download
LANGUAGES = ['de', 'es', 'fr', 'ru', 'tu', 'en']

SEED = 3407

INSTRUCTION_TEMPLATE = {
    "en": "Please summarize the following text in a few sentences, highlighting the most important points.",
    "es": "Por favor, resuma el siguiente texto en unas pocas frases, destacando los puntos más importantes.",
    "fr": "Veuillez résumer le texte suivant en quelques phrases, en mettant en évidence les points les plus importants.",
    "de": "Bitte fassen Sie den folgenden Text in ein paar Sätzen zusammen und heben Sie die wichtigsten Punkte hervor.",
    "ru": "Пожалуйста, подытожите следующий текст несколькими предложениями, выделив наиболее важные моменты.",
    "tu": "Lütfen aşağıdaki metni birkaç cümlede özetleyin ve en önemli noktaları vurgulayın."
}

def generate_training_prompt(
    system_prompt: str, document: str, summary: str
) -> str:
    summary = summary.replace("\n", "")

    return f"""### Instruction: {system_prompt}

### Input:
{document.strip()}

### Response:
{summary}
""".strip()


def setup_environment(args):
    warnings.filterwarnings("ignore")
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    if not args.wandb:
        os.environ["WANDB_DISABLED"] = "true"
    seed_everything(seed=SEED)
    torch.backends.cudnn.deterministic = True

def generate_names_for_wandb_run(model_name, dataset_name, epochs):
    model_name = model_name.split("/")[-1]
    dataset_name = dataset_name.split("/")[-1]
    return f"{model_name}-{dataset_name}-{epochs}"

if __name__ == '__main__':
    print("This is a utils file")