# PyTorch Lightning Module
import pytorch_lightning as pl
import torch

from embedding_distilation.Connector import Connector
from embedding_distilation.ProjectionHead import ProjectionHead
from embedding_distilation.reg import LocalRelationLoss

class EmbeddingAlignmentModel(pl.LightningModule):
    def __init__(self, large_embedding_dim, small_embedding_dim, projected_dim, inner_dim, lr=1e-3):
        super().__init__()
        self.projector = ProjectionHead(projected_dim, large_embedding_dim)
        self.connector = Connector(small_embedding_dim, projected_dim, inner_dim)
        self.lr = lr
        self.criterion = torch.nn.MSELoss() # L2 norm error
        #self.lrl = LocalRelationLoss()

    def forward(self, embedding_large, embedding_small):
        # Detach embeddings para romper la conexión con el grafo original
        embedding_large = embedding_large.detach()
        embedding_small = embedding_small.detach()

        # Normalización
        #embedding_large = embedding_large / (torch.norm(embedding_large, dim=-1, keepdim=True) + 1e-8)
        #embedding_small = embedding_small / (torch.norm(embedding_small, dim=-1, keepdim=True) + 1e-8)

        transformed_small = self.connector(embedding_small)
        projected_large = self.projector(transformed_small)
        return projected_large, transformed_small
    
    def calculate_loss(self, projected_large, embedding_large, transformed_small):
        loss = self.criterion(projected_large, embedding_large)
        #loss_reg = self.lrl(embedding_large, transformed_small)
        #return loss + loss_reg, loss, loss_reg
        return loss

    def training_step(self, batch, batch_idx):
        embedding_large, embedding_small = batch
        projected_large, transformed_small = self(embedding_large, embedding_small)
        loss = self.calculate_loss(projected_large, embedding_large, transformed_small)

        # Logging adicional
        similarity = torch.nn.functional.cosine_similarity(projected_large, embedding_large, dim=-1).mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("cosine_similarity", similarity, prog_bar=True)
        #self.log("mse_loss", loss_mse, prog_bar=True)
        #self.log("lrl_loss", loss_reg, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        embedding_large, embedding_small = batch
        projected_large,transformed_small = self(embedding_large, embedding_small)
        loss = self.calculate_loss(projected_large, embedding_large, transformed_small)

        # Logging adicional
        similarity = torch.nn.functional.cosine_similarity(projected_large, embedding_large, dim=-1).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("cosine_similarity_val", similarity, prog_bar=True)
        #self.log("mse_loss", loss_mse, prog_bar=True)
        #self.log("lrl_loss", loss_reg, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)