# Dummy Dataset
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_large, embeddings_small):
        self.embeddings_large = embeddings_large
        self.embeddings_small = embeddings_small

    def __len__(self):
        return len(self.embeddings_large)

    def __getitem__(self, idx):
        return self.embeddings_large[idx], self.embeddings_small[idx]