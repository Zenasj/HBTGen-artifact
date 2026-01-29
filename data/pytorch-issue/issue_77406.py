# torch.rand(4, dtype=torch.float32)  # Input tensor a of shape (4,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('indices', torch.tensor([3, 2, 1, 0]))
        self.register_buffer('values', torch.tensor([[0.], [1.], [2.], [3.]]))

    def forward(self, a):
        # Compute CPU result
        a_cpu = a.to('cpu').clone()
        try:
            cpu_result = a_cpu.index_put([self.indices], self.values, accumulate=True)
        except RuntimeError:
            cpu_result = None

        # Compute CUDA result
        cuda_result = None
        if torch.cuda.is_available():
            a_cuda = a.to('cuda').clone()
            indices_cuda = self.indices.to('cuda')
            values_cuda = self.values.to('cuda')
            cuda_result = a_cuda.index_put([indices_cuda], values_cuda, accumulate=True)

        # Compare results
        if cpu_result is not None and cuda_result is not None:
            return torch.allclose(cpu_result, cuda_result.to(cpu_result.device))
        else:
            return torch.tensor(False, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4)

