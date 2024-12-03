import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

# Dummy Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_large, embeddings_small):
        self.embeddings_large = embeddings_large
        self.embeddings_small = embeddings_small

    def __len__(self):
        return len(self.embeddings_large)

    def __getitem__(self, idx):
        return self.embeddings_large[idx], self.embeddings_small[idx]

# PyTorch Lightning Module
class EmbeddingAlignmentModel(pl.LightningModule):
    def __init__(self, large_embedding_dim, small_embedding_dim, projected_dim, lr=1e-3):
        super().__init__()
        self.projector = torch.nn.Linear(large_embedding_dim, projected_dim)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(small_embedding_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, projected_dim)
        )
        self.lr = lr
        self.criterion = torch.nn.MSELoss()

    def forward(self, embedding_large, embedding_small):
        # Detach embeddings para romper la conexión con el grafo original
        embedding_large = embedding_large.detach()
        embedding_small = embedding_small.detach()

        # Normalización
        # embedding_large = embedding_large / (torch.norm(embedding_large, dim=-1, keepdim=True) + 1e-8)
        # embedding_small = embedding_small / (torch.norm(embedding_small, dim=-1, keepdim=True) + 1e-8)

        projected_large = self.projector(embedding_large)
        transformed_small = self.connector(embedding_small)
        return projected_large, transformed_small

    def training_step(self, batch, batch_idx):
        embedding_large, embedding_small = batch
        projected_large, transformed_small = self(embedding_large, embedding_small)
        loss = self.criterion(projected_large, transformed_small)

        # Logging adicional
        similarity = torch.nn.functional.cosine_similarity(projected_large, transformed_small, dim=-1).mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("cosine_similarity", similarity, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

# Preparación de embeddings
def prepare_embeddings(tokenizer, model, sentences, max_length=32):
    embeddings = []
    for sentence in sentences:
        input_ids = tokenizer(sentence, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length").input_ids
        embedding = model.gpt_neox.embed_in(input_ids)
        embeddings.append(embedding.squeeze(0))  # Saca el batch dimension
    return torch.stack(embeddings)

# Main Training Script
if __name__ == "__main__":
    # Load Tokenizer and Models
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token
    model_small = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    model_large = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")

    # Clone embeddings
    embedding_small = copy.deepcopy(model_small.gpt_neox.embed_in)
    # quitar el gradiente
    embedding_small.weight.requires_grad = False
    embedding_large = copy.deepcopy(model_large.gpt_neox.embed_in)
    embedding_large.weight.requires_grad = False

    # Example sentences
    sentences = [
        "Hello, my dog is cute", 
        "The weather is nice today", 
        "PyTorch Lightning is awesome",
        "I love Transformers library"
        "I like to eat pizza"
        "My favorite color is blue"
    ]
    embeddings_large = prepare_embeddings(tokenizer, model_large, sentences)
    embeddings_small = prepare_embeddings(tokenizer, model_small, sentences)

    # Dataset and DataLoader
    dataset = EmbeddingDataset(embeddings_large, embeddings_small)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize Model
    model = EmbeddingAlignmentModel(
        large_embedding_dim=embeddings_large.shape[-1],
        small_embedding_dim=embeddings_small.shape[-1],
        projected_dim=128
    )

    # Training
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, dataloader)
