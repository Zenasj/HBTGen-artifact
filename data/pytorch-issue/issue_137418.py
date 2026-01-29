# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (B, 3072, 3072)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3072, 64)

    def forward(self, x):
        print("==========before call linear,---------")
        print(x)
        print(x.shape)
        print("-------------=========-------")
        result = self.linear(x)
        print("==========after call linear,---------")
        print(result)
        print(result.shape)
        return result

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 3072, 3072).cuda()

