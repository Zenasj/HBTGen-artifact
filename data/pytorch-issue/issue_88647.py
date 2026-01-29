# torch.rand(1, 1, 2, 2, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, 1, 2, 2, requires_grad=True))

    def forward(self, x):
        cpu_ok, cuda_ok = True, True
        out_cpu, out_cuda = None, None

        # Compute CPU version
        try:
            x_cpu = x.to('cpu')
            out_cpu = F.conv_transpose2d(x_cpu, self.weight.to('cpu'), stride=2, padding=2)
        except RuntimeError:
            cpu_ok = False

        # Compute CUDA version
        try:
            x_cuda = x.to('cuda')
            out_cuda = F.conv_transpose2d(x_cuda, self.weight.to('cuda'), stride=2, padding=2)
        except (RuntimeError, torch.cuda.CudaError):
            cuda_ok = False

        # Compare outcomes
        if cpu_ok != cuda_ok:
            return torch.tensor(True, dtype=torch.bool)
        elif cpu_ok and cuda_ok:
            # Check shape and values
            if out_cpu.shape != out_cuda.shape:
                return torch.tensor(True, dtype=torch.bool)
            try:
                # Compare on CPU for consistency
                cuda_on_cpu = out_cuda.to('cpu')
                if not torch.allclose(out_cpu, cuda_on_cpu, atol=1e-5):
                    return torch.tensor(True, dtype=torch.bool)
            except:
                return torch.tensor(True, dtype=torch.bool)
            return torch.tensor(False, dtype=torch.bool)
        else:
            return torch.tensor(False, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 2, 2, dtype=torch.float32)

