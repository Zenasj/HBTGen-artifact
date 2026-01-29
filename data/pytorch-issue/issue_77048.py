# torch.rand(1, 100, 64, 32, 32, dtype=torch.float)  # Inferred input shape

import torch
from torch.fft import fftn, fftshift
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.kernel1 = torch.rand(64, 32, 32, 30).cuda()
        self.kernel2 = torch.rand(64, 32, 32, 30).cuda()

    def forward(self, B):
        B = fftshift(B, dim=(-3, -2, -1))
        C = 0
        for i in range(30):
            f1 = B * self.kernel1[:, :, :, i]
            f2 = B * self.kernel2[:, :, :, i]
            f11 = fftn(f1, dim=(-3, -2, -1), norm="backward")
            f22 = fftn(f2, dim=(-3, -2, -1), norm="backward")
            C = C + f11 * f22
        return C

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    A = torch.rand(1, 1, 64, 32, 32)
    B = A.expand((1, 100, 64, 32, 32)).clone().cuda() / 64 / 32 / 32
    return B

