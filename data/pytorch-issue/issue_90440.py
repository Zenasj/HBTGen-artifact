import torch.nn as nn
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

device = torch.device('mps')

class MyLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size, input_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out

def train_step(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    train_losses = []
    for epoch in range(epochs):
        print("Epoch", epoch)
        train_loss = 0
        for x, y in train_loader:
            train_loss += train_step(model, criterion, optimizer, x, y)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print("Train loss:", train_loss)
    return train_losses

class MyDataset(Dataset):
    def __init__(self, df, window_size):
        self.df = df
        self.window_size = window_size
        self.data = []
        self.labels = []
        for i in range(len(df) - window_size):
            x = torch.tensor(df.iloc[i:i+window_size].values, dtype=torch.float, device=device)
            y = torch.tensor(df.iloc[i+window_size].values, dtype=torch.float, device=device)
            self.data.append(x)
            self.labels.append(y)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MyDataLoader(DataLoader):
    def __init__(self, dataset, window_size, batch_size, shuffle=True):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle)

df = pd.DataFrame(np.random.randint(0,100,size=(100, 1)))

model = MyLSTM(1, 1, 1, 1)
model.to(device)

train_data = MyDataset(df, 5)

train_loader = MyDataLoader(train_data, 5, 16)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = train_model(model, criterion, optimizer, train_loader, None, epochs=10)