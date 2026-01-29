import torch
import torch.nn as nn

# torch.rand(B, C, dtype=torch.float32)  # Inferred input shape: 2D tensor for Linear layer
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(100, 10)  # Example dimensions matching input shape

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    # Use XPU if available (fixed in PyTorch 2.7+ per issue resolution)
    if torch.xpu.is_available():
        model = model.to('xpu')
    return model

def GetInput():
    # Generate 2D tensor matching Linear layer's input requirements
    return torch.rand(1, 100, dtype=torch.float32)

