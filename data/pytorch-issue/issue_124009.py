# torch.rand(1, 2, dtype=torch.float32)  # Inferred input shape from tracing example
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Reconstructed based on input shape (2 features) and typical dense model structure
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

def my_model_function():
    # Initialize model with basic configuration
    model = MyModel()
    # Note: Actual weights would be loaded from the saved model state_dict in practice
    return model

def GetInput():
    # Generate input matching (1, 2) shape used in tracing
    return torch.rand(1, 2, dtype=torch.float32)

