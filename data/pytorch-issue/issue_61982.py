# torch.rand(B, 10, dtype=torch.float32)  # Inferred input shape based on example usage
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example layer to produce output tensor

    def forward(self, x):
        # Simulate model output that may be modified post-DDP
        return self.fc(x).sum()  # Returns scalar tensor for simplicity

def my_model_function():
    # Returns model instance with random initialization
    model = MyModel()
    return model

def GetInput():
    # Generate random input tensor matching the model's expected shape
    return torch.rand(4, 10, dtype=torch.float32)  # Batch size 4, 10 features

