import torch.nn as nn
import torchvision

import os

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet18


class LitResNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(512, 100)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        output = self.resnet(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self.resnet(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.resnet(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = CIFAR100(os.getcwd(), download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [45000, 5000])

model = LitResNet()
trainer = pl.Trainer(max_epochs=1, gpus=1)
trainer.fit(model,
            DataLoader(train, batch_size=64, num_workers=4),
            DataLoader(val, batch_size=64, num_workers=4))