# torch.randint(0, 5500, (50, 128), dtype=torch.int32) ← Add a comment line at the top with the inferred input shape
import torch
from collections import OrderedDict

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 300
        layers = [('embedding',
                   torch.nn.Embedding(5500, input_size))]

        transformerLayer = torch.nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=10,
            dim_feedforward=input_size,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        layers += [('transformerStack',
                   torch.nn.TransformerEncoder(transformerLayer, num_layers=2))]

        self.nn = torch.nn.Sequential(OrderedDict(layers))

    def forward(self, X):
        return self.nn(X)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 5500, (50, 128), dtype=torch.int32)

