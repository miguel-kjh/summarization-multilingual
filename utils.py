import os
import warnings

from accelerate import Accelerator, FullyShardedDataParallelPlugin
import pandas as pd
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
LANGUAGES = ['en', 'de', 'fr', 'ru', 'tu', 'es']

SEED = 3407

INSTRUCTION_TEMPLATE = {
    "en": "Please summarize the following text in a few sentences, highlighting the most important points.",
    "es": "Por favor, resuma el siguiente texto en unas pocas frases, destacando los puntos más importantes.",
    "fr": "Veuillez résumer le texte suivant en quelques phrases, en mettant en évidence les points les plus importants.",
    "de": "Bitte fassen Sie den folgenden Text in ein paar Sätzen zusammen und heben Sie die wichtigsten Punkte hervor.",
    "ru": "Пожалуйста, подытожите следующий текст несколькими предложениями, выделив наиболее важные моменты.",
    "tu": "Lütfen aşağıdaki metni birkaç cümlede özetleyin ve en önemli noktaları vurgulayın."
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


def setup_environment(args):
    warnings.filterwarnings("ignore")
    if not args.wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_PROJECT"] = PROJECT_NAME
        run_name = generate_names_for_wandb_run(args)
        wandb.init(project=PROJECT_NAME, entity="miguel_kjh", name=run_name)
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
    name_experiment = f"{model_name}-{dataset_name}-e{epochs}-b{batch_size}-lr{lr}-wd{weight_decay}-c{context}"
    name_experiment += f"-r{args.lora_r}-a{args.lora_alpha}-d{args.lora_dropout}" if args.lora else ""
    name_experiment += f"-nna{neftune_noise_alpha}" if neftune_noise_alpha is not None else ""
    return name_experiment

def create_accelerator():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    return Accelerator(fsdp_plugin=fsdp_plugin)


def create_model_and_tokenizer(args):
    if args.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
            bnb_4bit_use_double_quant= False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    accelerator = create_accelerator()
    model = accelerator.prepare_model(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def upload_to_wandb(table_name: str, summaries: list):
    df_original = pd.DataFrame(summaries)
    table_name = f"{wandb.run.name}-{table_name}"
    wandb.log(
        {
            table_name: wandb.Table(dataframe=df_original)
        }
    )
