import torch
import numpy as np

def _is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

# torch.rand(134217728, dtype=torch.float32, device='cuda')
class MyModel(torch.nn.Module):
    def __init__(self, sample_x):
        super().__init__()
        h_dim = int(sample_x.shape[0])
        assert _is_power_of_two(h_dim), "h_dim must be a power of two"
        self.h_dim_exp = int(np.log(h_dim) / np.log(2))
        self.working_shape = [1] + ([2] * self.h_dim_exp) + [1]

    def forward(self, x):
        result = x.view(self.working_shape)
        for i in range(self.h_dim_exp):
            dim = i + 1
            arrs = torch.chunk(result, 2, dim=dim)
            result = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), dim=dim)
        return result.view(x.shape)

def my_model_function():
    sample_x = torch.randn(134217728, device='cuda')
    return MyModel(sample_x)

def GetInput():
    return torch.randn(134217728, device='cuda', requires_grad=True)

