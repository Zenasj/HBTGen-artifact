import sys
from random import sample
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn as nn

from torchvision import transforms, datasets

import time


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out


def train(dataloader, model, loss_fn, optimizer, seq_dim, input_dim):
    size = len(dataloader.dataset)
    model.train()
    bt = time.time()
    run_time_list=[]
    delta_time = 0
    steps = 100

    for batch, (x, y) in enumerate(dataloader):
        x = x.view(-1, seq_dim, input_dim).requires_grad_()
        print("x.shape",x.shape)
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch > 0:
            delta_time = time.time()-bt
            run_time_list.append(delta_time)
        if batch % steps == 0 and batch>0:
            loss, current = loss.item(), (batch + 1) * len(x)
            if rank==0:
                print(f"No:{batch} loss: {loss:>7f}  [{current:>5d}/{size:>5d}] time={delta_time*1000:>0.1f}ms")

        bt = time.time()

    avg_time = sum(run_time_list)/len(run_time_list)*1000
    print(f"Train: avg_time={avg_time:>0.1f}ms loss={loss:>8f}")

    return loss




def main():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


    num_class = 10

    input_dim = 28
    seq_dim = 28
    hidden_dim = 100
    layer_dim = 1
    output_dim = num_class

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)


    print("create model")
    model.train()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 1
    bt_time = time.time()
    loss = 0.0
    for t in range(epochs):
        loss = train(train_dataloader, model, loss_fn, optimizer, seq_dim, input_dim)
        # test(test_dataloader, model, loss_fn, seq_dim, input_dim, device, rank)


    print(f"Total time: {time.time()-bt_time:>0.1f} loss={loss:>8f}")
    torch.save(model, 'best_weights.pth')
    print("save model weight to best_weights.pth")


if __name__ == '__main__':
    main()