import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

# 1. Import the necessary libraries

# 2. Define the custom dataset class
class CustomDataset(IterableDataset):
    def __iter__(self):
        return self

    def __next__(self):
        rows = np.random.randint(100, 300)
        columns = np.random.randint(200, 400)
        data = np.random.rand(rows, columns, 100)
        labels = np.random.randint(0, 17, [rows, columns])
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels)


# 3. Define the model architecture
class Conv1DModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_channels, 17)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x) # commenting this removes leak
        x = x.transpose(1, 2)
        x = self.relu(x)
        x = self.fc(x)
        return x


# Create the dataset and data loader
dataset = CustomDataset()

# Initialize the model
model = Conv1DModel(in_channels=100, out_channels=100, kernel_size=1, padding=0)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
model.to('cuda')

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataset):
        outputs = model(inputs.to('cuda'))
        loss = criterion(outputs.transpose(1, 2), targets.to('cuda'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {i} Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")