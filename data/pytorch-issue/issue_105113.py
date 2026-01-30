import torch.nn as nn

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1,2)
        self.pool = nn.AdaptiveMaxPool3d(output_size=[0, 0, 1], return_indices=True)

    def forward(self, x):
        x = self.fc(x)
        print('After fc', x.size())
        x = self.pool(x)
        
        return x

# Model instantiation
model = MyModel()

# Input definition
input_tensor = torch.rand([2, 2, 1, 1])

# Forward pass
model(input_tensor)  # No error

# Model compilation and forward pass
torch.compile(model)(input_tensor)  # RuntimeError