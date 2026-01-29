# torch.rand(B, 10, dtype=torch.float32)  # Assuming 2D input (batch_size, 10 features)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Example layer with 10 inputs and 5 outputs
        # Register forward hook to demonstrate the bug scenario
        self.linear.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        # Returns a tensor (non-None) to trigger the old bug
        return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=output.dtype)

    def forward(self, x):
        # Forward pass using the modified output from the hook
        return self.linear(x)

def my_model_function():
    # Returns the model instance with the hook setup
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape (B, 10)
    return torch.rand(2, 10)  # Batch size 2, 10 features

