# torch.rand(B, 1, 100, 375, 128, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 10, kernel_size=(3, 3, 10), stride=(1, 1, 1), padding=(1, 1, 0))

    def forward(self, x):
        return self.conv1(x)

def my_model_function():
    model = MyModel()
    model.cuda()  # Matches the original issue's CUDA context
    return model

def GetInput():
    return torch.randn(25, 1, 100, 375, 128, device='cuda')  # Fails at batch size 25 as described

