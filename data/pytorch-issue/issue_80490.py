# torch.rand(1, 4, dtype=torch.float32)  # Inferred input shape: (batch_size, input_features)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.register_buffer("buf", torch.randn(4))

    def forward(self, x):
        x = self.fc(x)
        x = x + self.buf
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 4, dtype=torch.float32)

