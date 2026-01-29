# torch.rand(B, C, H, W, dtype=torch.complex128)  # Assuming a typical input shape for a complex-valued model

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size).to(torch.complex128)
        self.fc2 = nn.Linear(self.hidden_size, 1).to(torch.complex128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden.real) + 1j * self.relu(hidden.imag)
        output = self.fc2(relu)
        output = self.sigmoid(output.real) + 1j * self.sigmoid(output.imag)
        return output

    def relu(self, x):
        return torch.clamp(x, min=0)

def my_model_function():
    return MyModel(input_size=10, hidden_size=5)

def GetInput():
    B, C, H, W = 1, 1, 1, 10  # Example batch size, channels, height, width
    input_tensor = torch.rand(B, C, H, W, dtype=torch.complex128)
    return input_tensor

