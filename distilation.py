import copy
import os
import pytorch_lightning as pl
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from lightning.pytorch.loggers import WandbLogger
import wandb


from utils import prepare_embeddings, SEED, wandb_end
from embedding_distilation.EmbeddingDataset import EmbeddingDataset
from embedding_distilation.EmbeddingAlignmentModel import EmbeddingAlignmentModel

PROJECT_NAME = "embedding-distilation"
FOLDER_TO_SAVE = "models/connetor"
os.makedirs(FOLDER_TO_SAVE, exist_ok=True)

CONTEXT     = 512
BATCH_SIZE  = 4
INNER_DIM   = 512
MAX_EPOCHS  = 10
LARGE_MODEL = "EleutherAI/pythia-160m"
SMALL_MODEL = "EleutherAI/pythia-14m"
TOKENIZER   = "EleutherAI/pythia-14m"
DATASET     = "data/03-combined/tiny"
WANDB       = False

def setup_wandb():
    wandb.init(project=PROJECT_NAME, entity="miguel_kjh", name="test")
    wandb_logger = WandbLogger()
    return wandb_logger

# Main Training Script
if __name__ == "__main__":
    
    pl.seed_everything(SEED)
    logger = setup_wandb() if WANDB else None
    # Load Tokenizer and Models
    tokenizer           = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token
    model_small         = AutoModelForCausalLM.from_pretrained(SMALL_MODEL)
    model_large         = AutoModelForCausalLM.from_pretrained(LARGE_MODEL)

    # Clone embeddings
    embedding_small = copy.deepcopy(model_small.gpt_neox.embed_in)
    embedding_large = copy.deepcopy(model_large.gpt_neox.embed_in)

    # Example sentences
    dataset   = load_from_disk(DATASET)
    train_sentences = dataset['train']['text']
    embeddings_large = prepare_embeddings(tokenizer, model_large, train_sentences, context=CONTEXT)
    embeddings_small = prepare_embeddings(tokenizer, model_small, train_sentences, context=CONTEXT)
    val_sentences = dataset['validation']['text']
    embeddings_large_val = prepare_embeddings(tokenizer, model_large, val_sentences, context=CONTEXT)
    embeddings_small_val = prepare_embeddings(tokenizer, model_small, val_sentences, context=CONTEXT)

    # Dataset and DataLoader
    dataset        = EmbeddingDataset(embeddings_large, embeddings_small)
    dataloader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    dataset_val    = EmbeddingDataset(embeddings_large_val, embeddings_small_val)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = EmbeddingAlignmentModel(
        large_embedding_dim=embeddings_large.shape[-1],
        small_embedding_dim=embeddings_small.shape[-1],
        projected_dim=embeddings_small.shape[-1],
        inner_dim=INNER_DIM
    )

    # Training
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
    )
    trainer.fit(model, dataloader, dataloader_val)

    # Save Model only connector
    connector = model.connector
    torch.save(
        connector.state_dict(), 
        os.path.join(FOLDER_TO_SAVE, "connector.pth")
    )

    if WANDB:
        wandb_end()
