# torch.rand(128, 1024), torch.rand(128, 1024) ← input and hidden tensors
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.randn(input_size + hidden_size, 3 * hidden_size))
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))
    
    def forward(self, inputs):
        input, hidden = inputs
        update, reset = torch.chunk(
            torch.sigmoid(
                torch.addmm(
                    self.bias[:2 * self.hidden_size],
                    torch.cat([input, hidden], dim=1),
                    self.weight[:, :2 * self.hidden_size]
                )
            ),
            2, dim=1
        )
        candidate = torch.tanh(
            torch.addmm(
                self.bias[2 * self.hidden_size:],
                torch.cat([input, reset * hidden], dim=1),
                self.weight[:, 2 * self.hidden_size:]
            )
        )
        return update * hidden + (1 - update) * candidate

def my_model_function():
    return MyModel(input_size=1024, hidden_size=1024)

def GetInput():
    input = torch.rand(128, 1024, dtype=torch.float32)
    hidden = torch.rand(128, 1024, dtype=torch.float32)
    return (input, hidden)

