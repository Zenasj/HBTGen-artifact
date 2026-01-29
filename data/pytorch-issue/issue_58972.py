# torch.rand(B, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    class FixedSequential(nn.Sequential):
        def __init__(self, *args):
            super(MyModel.FixedSequential, self).__init__(*args)
            self._initial_modules = list(self._modules.keys())  # Capture initial modules

        def forward(self, input):
            # Only process the initial modules, ignoring dynamically added ones
            for name in self._initial_modules:
                module = self._modules[name]
                input = module(input)
            return input

    def __init__(self):
        super(MyModel, self).__init__()
        self.seq = MyModel.FixedSequential(nn.Linear(1, 1))  # Base model
        # Add an extra module to test the fix (should be ignored in forward)
        self.seq.linear2 = nn.Linear(2, 2)  # Dynamically added module (not part of forward)

    def forward(self, x):
        return self.seq(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, dtype=torch.float32)  # Shape (B=1, in_features=1)

