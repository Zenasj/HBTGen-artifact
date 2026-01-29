# torch.rand(B, C, H, W, dtype=...)  # Input shape: (1, 1, 2, 2)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Perform max pooling
        pooled, indices = self.max_pool(x)
        
        # Perform max unpooling
        unpooled = self.max_unpool(pooled, indices)
        
        return unpooled

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand([1, 1, 2, 2], dtype=torch.float32)

