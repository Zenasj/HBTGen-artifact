# torch.rand(1, 7, dtype=torch.float32)  # Inferred input shape (batch_size=1, features=7)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input: int, units: int) -> None:
        super().__init__()
        assert units > 0, 'units must be greater than 0'
        assert input > 0, 'input must be greater than 0'
        self.neural_network = nn.ModuleList()
        self.neural_network.append(nn.Linear(input, units))
        self.neural_network.append(nn.ReLU())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for layer in self.neural_network:
            tensor = layer(tensor)
        return tensor

def my_model_function():
    return MyModel(input=7, units=1)

def GetInput():
    return torch.rand(1, 7, dtype=torch.float32)

