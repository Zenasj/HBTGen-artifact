# torch.rand(B, C, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute both log_softmax and log(softmax) to compare gradients
        log_softmax_out = torch.log_softmax(x, dim=1)
        softmax_log_out = torch.softmax(x, dim=1).log()
        return log_softmax_out, softmax_log_out  # Return both outputs for gradient comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Return a uniform tensor (all ones) as in the issue's example to trigger the problem
    return torch.ones([2, 300], requires_grad=True).float()

