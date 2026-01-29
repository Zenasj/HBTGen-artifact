# torch.rand(B, 10, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        return self.fc(x).mean()  # Return scalar value for logging

def my_model_function():
    model = MyModel()
    # Initialize weights to avoid undefined parameters
    with torch.no_grad():
        model.fc.weight.fill_(0.1)
        model.fc.bias.fill_(0.1)
    return model

def GetInput():
    B = 1  # Batch size inferred from example loop
    return torch.rand(B, 10, 1, 1, dtype=torch.float32)

