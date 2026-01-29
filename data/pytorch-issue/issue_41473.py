# torch.rand(3, 6, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = torch.tensor(
            [
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 1],
            ],
            dtype=torch.bool,
        )

    def forward(self, x):
        # Compute CPU result
        x_cpu = x.to("cpu")
        permuted_cpu = x_cpu.permute(2, 0, 1)
        mask_cpu = self.mask.to("cpu")
        cpu_result = torch.masked_select(permuted_cpu, mask_cpu)

        # Compute GPU result if available
        if torch.cuda.is_available():
            x_gpu = x.to("cuda")
            permuted_gpu = x_gpu.permute(2, 0, 1)
            mask_gpu = self.mask.to("cuda")
            gpu_result = torch.masked_select(permuted_gpu, mask_gpu)
            gpu_result = gpu_result.cpu()
        else:
            return torch.tensor(False)  # GPU not available, cannot compare

        # Compare results
        return torch.all(cpu_result == gpu_result)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 6, 2, dtype=torch.float32)

