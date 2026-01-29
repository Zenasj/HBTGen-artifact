# torch.rand(NUM_INPUT, INPUT_SIZE, dtype=torch.float32, device=DEVICE)  # Inferred input shape

import torch
from torch import nn

NUM_INPUT = 50
INPUT_SIZE = 500
NUM_LINEAR = 2
DEVICE = "cuda"

class MyModel(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(NUM_INPUT, INPUT_SIZE // NUM_LINEAR)) for _ in range(NUM_LINEAR)]
        ).to(self.device)
        self.biases = torch.nn.ParameterList(
            [torch.randn(NUM_INPUT).to(self.device) for _ in range(NUM_LINEAR)]
        ).to(self.device)

    def to_float(self):
        for layer in [self.weights, self.biases]:
            layer = [param.cpu().float().to(self.device) for param in layer]

    def to_double(self):
        for layer in [self.weights, self.biases]:
            layer = [param.cpu().double().to(self.device) for param in layer]

    def forward(self, x):
        l1_out = torch.split(x.to(self.device), INPUT_SIZE // NUM_LINEAR, dim=1)
        l1_linear = []
        for i in range(len(l1_out)):
            l1_linear.append(
                torch.nn.functional.linear(
                    l1_out[i], self.weights[i], self.biases[i]
                )
            )
        l1_out = torch.cat(l1_linear, dim=1)
        return l1_out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    arg = torch.randn(NUM_INPUT, INPUT_SIZE, device=DEVICE)
    arg = arg + torch.randn(NUM_INPUT, INPUT_SIZE, device=DEVICE) + torch.tensor(100, dtype=torch.float32, device=DEVICE)
    return arg

