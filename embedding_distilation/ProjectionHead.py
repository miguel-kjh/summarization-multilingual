import torch


class ProjectionHead(torch.nn.Module):
    def __init__(self, large_embedding_dim, projected_dim):
        super().__init__()
        self.projector = torch.nn.Linear(large_embedding_dim, projected_dim)

    def forward(self, embedding_large):
        return self.projector(embedding_large)