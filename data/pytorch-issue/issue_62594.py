# torch.rand(2, 64, 50, 50, dtype=torch.float16) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    model.eval()  # Set the model to evaluation mode
    model.cuda()  # Move the model to GPU
    model.to(torch.float16)  # Convert the model to half precision
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    dtype = torch.float16
    device = torch.device('cuda')
    inp = torch.randn(2, 64, 50, 50, device=device, dtype=dtype)
    return inp

