import torch.nn as nn
import torchvision

train_dataloader = DataLoader(train_dataset, sampler=SubsetRandomSampler(train_mask))
val_dataloader = DataLoader(train_dataset, sampler=SubsetRandomSampler(val_mask))

train_dataset1, val_dataset1 = Subset(train_dataset, train_mask), Subset(train_dataset, val_mask)
train_dataloader = DataLoader(train_dataset1, shuffle=True)
val_dataloader = DataLoader(val_dataset1, shuffle=True)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

from torch.utils.data import Subset, SubsetRandomSampler
train_mask = range(50000)
val_mask = range(50000, 60000)

train_dataloader = DataLoader(train_dataset, sampler=SubsetRandomSampler(train_mask))
val_dataloader = DataLoader(train_dataset, sampler=SubsetRandomSampler(val_mask))

# Swap the above with this to drastically improve performance
# train_dataset1, val_dataset1 = Subset(train_dataset, train_mask), Subset(train_dataset, val_mask)
# train_dataloader = DataLoader(train_dataset1, shuffle=True)
# val_dataloader = DataLoader(val_dataset1, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

train(train_dataloader, model, loss_fn, optimizer)
test(val_dataloader, model, loss_fn)