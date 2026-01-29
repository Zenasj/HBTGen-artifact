# torch.rand(50, 10, 256, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.GRU(input_size=256,
                          hidden_size=128,
                          num_layers=2,
                          batch_first=False,
                          dropout=0.1,
                          bidirectional=True)

    def forward(self, x):
        return self.rnn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    model.eval()
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand((50, 10, 256), dtype=torch.float32)

