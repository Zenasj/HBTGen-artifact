# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fully_connected_1 = nn.Linear(input_size, hidden_size)
        self.early_exit_head_1 = nn.Linear(hidden_size, output_size)
        self.last_exit = nn.Linear(hidden_size, output_size)
        self.threshold = torch.tensor([0.0], dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        mean = x.mean()
        x = self.fully_connected_1(x)
        x = torch.cond(mean > 0.0, lambda: self.early_exit_head_1(x), lambda: self.last_exit(x))
        x = torch.cat([x, mean.reshape_as(x)], dim=1)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(input_size=1, hidden_size=3, output_size=1)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming B (batch size) = 1, C (channels) = 1, H (height) = 1, W (width) = 1
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

