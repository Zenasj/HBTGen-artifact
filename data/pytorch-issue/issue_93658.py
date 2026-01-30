import torch.nn.functional as F
import torchvision

import torchdynamo
from torchdynamo import optimize
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time
import torchmetrics
import pytorch_lightning as pl
import kornia as K

import numpy as np
from PIL import Image

"""## Define Data Augmentations module"""

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""
    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self._max_val: float = 255.

        self.transforms = nn.Sequential(
            K.enhance.Normalize(0., self._max_val),
            K.augmentation.RandomHorizontalFlip(p=0.5)
        )

        self.jitter = K.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

"""## Define a Pre-processing model"""

class PreProcess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self) -> None:
        super().__init__()
 
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float()

"""## Define PyTorch Lightning model"""

class CoolSystem(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(3 * 32 * 32, 10)

        self.preprocess = PreProcess()

        self.transform = DataAugmentation()

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        x_aug = self.transform(x)  # => we perform GPU/Batched data augmentation
        logits = self.forward(x_aug)
        loss = F.cross_entropy(logits, y)
        self.log('train_acc_step', self.accuracy(logits.argmax(1), y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        logits = self.forward(x)
        self.log('val_acc_step', self.accuracy(logits.argmax(1), y))
        return F.cross_entropy(logits, y)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.0004)

    def prepare_data(self):
        CIFAR10(os.getcwd(), train=True, download=True, transform=self.preprocess)
        CIFAR10(os.getcwd(), train=False, download=True, transform=self.preprocess)

    def train_dataloader(self):
        # REQUIRED
        dataset = CIFAR10(os.getcwd(), train=True, download=False, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=32)
        return loader

    def val_dataloader(self):
        dataset = CIFAR10(os.getcwd(), train=True, download=False, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=32)
        return loader

"""## Run training"""

from pytorch_lightning import Trainer

# init model
model = CoolSystem()

# Initialize a trainer
num_gpus: int = 0  # change to 1
trainer = Trainer(gpus=1, max_epochs=1)

tic = time.time()

# Train the model ⚡
# trainer.fit(model)
toc = time.time()

duration = toc - tic
print(f"Total running time without dynamo was {duration}")


tic = time.time()

@torchdynamo.optimize("inductor")
def wrapper_func():
    trainer.fit(model)

# Train the model ⚡
wrapper_func()
toc = time.time()

duration = toc - tic
print(f"Total running time with dynamo was {duration}")