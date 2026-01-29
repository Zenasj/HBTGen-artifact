# torch.rand(B, 24, 224) and torch.rand(B, 8, 224, 224)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1d submodule from the issue's test setup
        self.conv1d = nn.Sequential(
            nn.Conv1d(24, 256, kernel_size=(12,), stride=(6,), groups=4),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=(6,), stride=(3,), padding=(2,), groups=4),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), groups=4),
            nn.ReLU(),
        )
        # Conv2d submodule from the issue's test setup
        self.conv2d = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
    
    def forward(self, inputs):
        # Process both models in parallel and return outputs as tuple
        x1, x2 = inputs
        return (self.conv1d(x1), self.conv2d(x2))

def my_model_function():
    # Returns the fused model containing both convolution networks
    return MyModel()

def GetInput():
    # Returns tuple of inputs for both submodels (64 is batch size from issue's test)
    input1 = torch.rand(64, 24, 224, dtype=torch.float32)  # Conv1d input
    input2 = torch.rand(64, 8, 224, 224, dtype=torch.float32)  # Conv2d input
    return (input1, input2)

