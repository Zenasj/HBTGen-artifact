# torch.rand(B, 100, 1, 1, dtype=torch.float32)  # Input shape: batch_size x in_features x 1 x 1 (flattened to 2D in forward)

import torch
import torch.nn as nn
from torch.nn import ModuleList

class MyModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers=[5000, 1000, 500, 100], dropout=0.5, activation='relu'):
        super(MyModel, self).__init__()
        
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'prelu': nn.PReLU()
        }
        
        self.layers = ModuleList()
        
        # Initial layer: in_features → hidden_layers[0]
        self.layers.append(nn.Linear(in_features, hidden_layers[0]))
        self.layers.append(nn.BatchNorm1d(hidden_layers[0]))
        self.layers.append(activation_functions[activation])
        self.layers.append(nn.Dropout(dropout))
        
        # Intermediate layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.BatchNorm1d(hidden_layers[i+1]))
            self.layers.append(activation_functions[activation])
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], out_features))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape 4D input (B,C,1,1) → 2D (B,C)
        for layer in self.layers:
            x = layer(x)
        return x

def my_model_function():
    # Default parameters matching the inferred input shape (100 features)
    return MyModel(in_features=100, out_features=1)

def GetInput():
    # Returns a 4D tensor (B, C, 1, 1) compatible with the model's input expectation
    return torch.rand(1, 100, 1, 1, dtype=torch.float32)

