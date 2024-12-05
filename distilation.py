import argparse
import copy
import os
import random
import pytorch_lightning as pl
from datasets import load_from_disk
from distutils.util import strtobool
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from lightning.pytorch.loggers import WandbLogger
import wandb
import warnings
import json

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from utils import prepare_embeddings, SEED, wandb_end, get_timestamp
from embedding_distilation.EmbeddingDataset import EmbeddingDataset
from embedding_distilation.EmbeddingAlignmentModel import EmbeddingAlignmentModel

def generate_names_for_wandb_run(args: dict) -> str:
    large_model = args.large_model.split("/")[-1]
    small_model = args.small_model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    epochs = args.max_epochs
    batch_size = args.batch_size
    context = args.context
    inner_dim = args.inner_dim
    data_porcentage = args.data_porcentage
    time_now = get_timestamp()
    return f"{large_model}-{small_model}-{dataset_name}-e{epochs}-b{batch_size}-c{context}-id{inner_dim}-dp{data_porcentage}-{time_now}"

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Embedding distillation script")
    parser.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--context", type=int, default=512, help="Maximum context size for embeddings")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--inner_dim", type=int, default=512, help="Inner dimension for the alignment model")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of epochs for training")
    parser.add_argument("--large_model", type=str, default="EleutherAI/pythia-160m", help="Large model path")
    parser.add_argument("--small_model", type=str, default="EleutherAI/pythia-14m", help="Small model path")
    parser.add_argument("--dataset", type=str, default="data/03-combined/tiny", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="models/connectors", help="Directory to save the model")
    parser.add_argument("--project_name", type=str, default="embedding-distilation", help="WandB project name")
    parser.add_argument("--data_porcentage", type=float, default=0.1, help="Porcentage of data to use")
    args = parser.parse_args()

    time_now = get_timestamp()
    dataset_name = args.dataset.split("/")[-1]
    args.run_name = f"{args.large_model.split('/')[-1]}-{args.small_model.split('/')[-1]}-{dataset_name}-e{args.max_epochs}-b{args.batch_size}-c{args.context}-id{args.inner_dim}-{time_now}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def setup_wandb(args: dict) -> WandbLogger:
    wandb.init(project=args.project_name, entity="miguel_kjh", name=args.run_name)
    wandb_logger = WandbLogger()
    return wandb_logger

def save_model_connector(model: pl.LightningModule, folder: str, name_model: str = "connector.pth"):
    connector = model.connector
    connecto_folder = os.path.join(folder, name_model)
    os.makedirs(os.path.join(folder, name_model), exist_ok=True)
    torch.save(connector.state_dict(), os.path.join(connecto_folder, "weights.pth"))
    with open(os.path.join(connecto_folder, "config.json"), "w") as f:
        json.dump(connector.get_hyperparameters(), f, indent=4)


def main(args: dict):
    pl.seed_everything(SEED)
    logger = setup_wandb(args) if args.wandb else None
    # Load Tokenizer and Models
    tokenizer           = AutoTokenizer.from_pretrained(args.small_model)
    tokenizer.pad_token = tokenizer.eos_token
    model_small         = AutoModelForCausalLM.from_pretrained(args.small_model)
    model_large         = AutoModelForCausalLM.from_pretrained(args.large_model)

    # Dataset and Embeddings
    dataset   = load_from_disk(args.dataset)
    train_sentences = dataset['train']['text']
    train_sentences = random.sample(train_sentences, len(train_sentences))
    train_sentences = train_sentences[:int(len(train_sentences) * args.data_porcentage)]
    # TODO: improve prepare_embeddings to generate embeddings for all text
    embeddings_large = prepare_embeddings(tokenizer, model_large, train_sentences, context=args.context)
    embeddings_small = prepare_embeddings(tokenizer, model_small, train_sentences, context=args.context)
    val_sentences = dataset['validation']['text']
    embeddings_large_val = prepare_embeddings(tokenizer, model_large, val_sentences, context=args.context)
    embeddings_small_val = prepare_embeddings(tokenizer, model_small, val_sentences, context=args.context)

    # Dataset and DataLoader
    dataset        = EmbeddingDataset(embeddings_large, embeddings_small)
    dataloader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataset_val    = EmbeddingDataset(embeddings_large_val, embeddings_small_val)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize Model
    model = EmbeddingAlignmentModel(
        large_embedding_dim=embeddings_large.shape[-1],
        small_embedding_dim=embeddings_small.shape[-1],
        projected_dim=embeddings_small.shape[-1],
        inner_dim=args.inner_dim,
    )

    # Training
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
    )
    trainer.fit(model, dataloader, dataloader_val)

    # Save Model only connector
    save_model_connector(model, args.output_dir, f"connector_{args.run_name}")

    if args.wandb:
        wandb_end()

# Main Training Script
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
