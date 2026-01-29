import torch
from torch import nn

# torch.rand(4, 8, 32, 32, dtype=torch.float, device='cuda')
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.ConvTranspose2d(8, 16, kernel_size=3, padding=[3, 3])  # Faulty list-based padding

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    model = model.cuda()  # Move model to CUDA as in the original example
    return model

def GetInput():
    return torch.randn(4, 8, 32, 32, device='cuda', dtype=torch.float)  # Matches input shape and device from issue

