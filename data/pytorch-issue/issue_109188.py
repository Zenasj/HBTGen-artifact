# torch.rand(2, 12, dtype=torch.bfloat16)
import torch
import copy

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_cpu = torch.nn.LayerNorm(12, eps=0.0, elementwise_affine=False).bfloat16()
        self.ln_cuda = copy.deepcopy(self.ln_cpu).to('cuda')  # Ensure identical initial weights

    def forward(self, x):
        # Compute outputs on both devices
        out_cpu = self.ln_cpu(x.to('cpu'))
        out_cuda = self.ln_cuda(x.to('cuda'))
        # Compare outputs on CPU (to avoid GPU-CPU sync issues)
        return torch.abs(out_cpu - out_cuda.to('cpu')).max()  # Return max difference tensor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 12, dtype=torch.bfloat16)

