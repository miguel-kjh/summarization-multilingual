import os
from typing import Dict
import json
import torch


class Connector(torch.nn.Module):
    def __init__(self, small_embedding_dim, projected_dim, inner_dim):
        super().__init__()
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(small_embedding_dim, inner_dim),
            torch.nn.GELU(),
            torch.nn.Linear(inner_dim, projected_dim)
        )

    def get_hyperparameters(self) -> Dict[str, int]:
        return {
            "small_embedding_dim": self.connector[0].in_features,
            "projected_dim": self.connector[-1].out_features,
            "inner_dim": self.connector[0].out_features,
        }
    
    @staticmethod
    def from_pretraining(folder: str) -> "Connector":
        with open(os.path.join(folder, "config.json"), "r") as f:
            hyperparameters = json.load(f)
        weight_path = os.path.join(folder, "weights.pth")
        connector = Connector(
            small_embedding_dim=hyperparameters["small_embedding_dim"],
            projected_dim=hyperparameters["projected_dim"],
            inner_dim=hyperparameters["inner_dim"],
        )
        connector.load_state_dict(torch.load(weight_path))
        return connector

    def forward(self, embedding_small) -> torch.Tensor:
        return self.connector(embedding_small)