import torch
import torch.nn as nn

# torch.rand(1, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.doubler = nn.Linear(1, 1, bias=False)
        self.doubler.weight.data.fill_(2.0)  # Fixed weight to double the input
        self.register_buffer('counter', torch.tensor(0, dtype=torch.int32))  # Tracks state for multiplier_counter

    def forward(self, x):
        # Doubler's output (2*x)
        doubled = self.doubler(x)
        
        # MultiplierCounter's current value (k*x where k is counter)
        current_counter = self.counter
        multiplied = current_counter * x  # Current value from counter
        
        # Compare outputs using torch.allclose with a tolerance (assumed from examples)
        # Returns False if outputs are different (as tensors)
        are_different = torch.any(doubled != multiplied)
        
        # Increment counter for next call (mimics generator state progression)
        self.counter += 1
        
        return are_different  # Returns boolean tensor indicating difference

def my_model_function():
    return MyModel()

def GetInput():
    # Random input matching the (1,) shape from examples
    return torch.rand(1, dtype=torch.float32)

