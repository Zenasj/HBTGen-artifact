import torch.nn as nn

# import dependencies
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# load data
df = pd.read_csv('data/simsolid.csv')

# split data into features and labels
x = df.drop(columns=['run', 'accuracy']).values
y = df['accuracy'].values

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# layer details
units = 1
input_shape = x.shape[1]
batch_size = 10

# create model
class model(torch.nn.Module):
    """
    Define the neural network architecture

    Parameters
    ----------
    input : int
        number of input features
    units : int
        number of units in the hidden layer

    Returns
    -------
    tensor: torch.Tensor
        output tensor of the neural network
    """
    def __init__(self, input: int, units: int) -> None:
        super().__init__()
        assert units > 0, 'units must be greater than 0'
        assert input > 0, 'input must be greater than 0'

        self.neural_network = torch.nn.ModuleList()
        self.neural_network.add_module('layer_1', torch.nn.Linear(input, units))
        self.neural_network.add_module('activation_1', torch.nn.ReLU())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for layer in self.neural_network:
            tensor = layer(tensor)
        return tensor

net = model(input_shape, units)

# define loss function
creterian = torch.nn.MSELoss()

# define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# train model
epochs = 100

# log the training 
path = 'logs/training.log'
status = os.path.exists(path)
if status:
    mode = 'w'
else:
    mode = 'x'

with open(path, mode) as log_file:
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = torch.tensor(np.reshape(x_train[i:i+batch_size], (batch_size, input_shape))).float()
            y_batch = torch.tensor(np.reshape(y_train[i:i+batch_size], (batch_size, 1))).float()
            y_pred = net(x_batch)
            loss = creterian(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_file.write(f'Epoch: {epoch}, Loss: {loss.item()}\n')

# score model
tensor = torch.tensor([2, 16, 0.145, 6, 0, 1, 2]).float()
print(net(tensor))

# save model as torch script
path = 'models/dummy_model.pt'
torch.jit.save(torch.jit.script(net), path)