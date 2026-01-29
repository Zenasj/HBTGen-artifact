# torch.rand(B, 1, 28, 28, dtype=torch.float32)
import torch
from torch import nn
from torchmetrics import Accuracy
import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )
    
    def forward(self, x):
        embedding = self.encoder(x.view(x.size(0), -1))
        return embedding
    
    def training_step(self, batch, batch_idx):
        # Problematic code causing the error (as per the user's test snippet)
        device = self.device
        num_samples = 1000
        num_classes = 34
        Y = torch.ones(num_samples, dtype=torch.long, device=device)
        X = torch.zeros(num_samples, num_classes, device=device)
        accuracy = Accuracy(average="none", num_classes=num_classes).to(device)
        accuracy(X, Y)  # Triggers computation during step
        
        # Original autoencoder training logic
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.MSELoss()(x_hat, x)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 1, 28, 28, dtype=torch.float32)

