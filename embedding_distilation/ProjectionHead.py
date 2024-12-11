import torch
import torch.nn as nn

class ProjectionHead(torch.nn.Module):
    def __init__(self, projected_dim, large_embedding_dim):
        super().__init__()
        print(f"Projected Dim: {projected_dim}")
        print(f"Large Embedding Dim: {large_embedding_dim}")
        self.projector = nn.Sequential(
            nn.Linear(projected_dim, projected_dim*2),
            nn.GELU(),
            nn.Linear(projected_dim*2, large_embedding_dim),
        )

    def forward(self, embedding_small):
        return self.projector(embedding_small)