import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, roc_auc_score

from utils import SEED

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        self.samples = torch.tensor(data["sample"], dtype=torch.float32)
        # normalize the samples
        self.samples = (self.samples - self.samples.mean()) / self.samples.std()
        self.labels = torch.tensor(data["label"], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# Define MLP model
class MLPModel(pl.LightningModule):
    def __init__(self, input_size, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        preds = self(x)
        loss = self.criterion(preds, y)
        preds_binary = (preds > 0.5).float()
        accuracy = (preds_binary == y).float().mean()
        f1 = f1_score(y.cpu(), preds_binary.cpu(), zero_division=1)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        preds = self(x)
        loss = self.criterion(preds, y)
        preds_binary = (preds > 0.5).float()
        accuracy = (preds_binary == y).float().mean()
        f1 = f1_score(y.cpu(), preds_binary.cpu(), zero_division=1)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        preds = self(x)
        loss = self.criterion(preds, y)
        preds_binary = (preds > 0.5).float()
        accuracy = (preds_binary == y).float().mean()
        f1 = f1_score(y.cpu(), preds_binary.cpu(), zero_division=1)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", accuracy, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

# Main function
def main():
    pl.seed_everything(SEED)
    # Paths to datasets
    dataset_train_path = "data/02-processed/spanish/clusters_clf_test.pkl"
    dataset_validation_path = "data/02-processed/spanish/clusters_clf_train.pkl"
    dataset_test_path = "data/02-processed/spanish/clusters_clf_validation.pkl"

    # Load datasets
    train_dataset = CustomDataset(dataset_train_path)
    val_dataset = CustomDataset(dataset_validation_path)
    test_dataset = CustomDataset(dataset_test_path)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=31)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=31)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=31)

    # Get input size from dataset
    input_size = train_dataset.samples.shape[1]

    # Initialize model
    model = MLPModel(input_size=input_size)

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="mlp-best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # Trainer
    trainer = Trainer(
        max_epochs=10,
        #callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    # Save the final model
    #torch.save(model.state_dict(), "mlp_model.pt")

if __name__ == '__main__':
    main()
