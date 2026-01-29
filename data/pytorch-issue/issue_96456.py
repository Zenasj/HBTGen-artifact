# torch.rand(1, 12, 256*64, requires_grad=True) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.scale_factor = 0.5

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (256, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.relu()
        x = self.transpose_for_scores(x)
        x /= torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float) * self.scale_factor)
        return x.transpose(-1, -2)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand((1, 12, 256*64), requires_grad=True)

