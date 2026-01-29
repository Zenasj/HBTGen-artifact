# torch.rand(1, 3, 16, 16, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1, 1)
        self.conv.register_full_backward_hook(self.hook_out)

    def hook_out(self, module, grad_input, grad_output):
        print("backward hook out")

        def hook_in(module, grad_in, grad_out):
            print("backward hook in")

        # Reproduce problematic tensor operations inside backward hook
        inp = torch.eye(4, 5, requires_grad=True)
        out = (inp + 1).pow(2).t()
        print(out)
        # This line triggers the error described in the issue
        out.backward(torch.ones_like(out), retain_graph=True)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 16, 16, dtype=torch.float32)

