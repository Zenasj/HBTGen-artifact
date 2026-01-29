# torch.rand(B, 17, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_cuda = torch.randn(10, 15, device='cuda')  # Matrix multiplication example tensors
        self.B_cuda = torch.randn(14, 32, device='cuda')
        self.linear_cuda = nn.Linear(16, 32).cuda()      # Linear layer example
        self.linear_cpu = nn.Linear(16, 32)              # CPU baseline

    def forward(self, x):
        # Compare matrix multiplication behavior between CUDA and CPU
        try:
            cuda_matmul = self.A_cuda @ self.B_cuda
        except:
            cuda_matmul = None
        try:
            cpu_matmul = self.A_cuda.cpu() @ self.B_cuda.cpu()
        except RuntimeError:
            cpu_matmul = None
        mat_discrepancy = (cuda_matmul is not None) and (cpu_matmul is None)

        # Compare linear layer behavior between CUDA and CPU
        try:
            cuda_output = self.linear_cuda(x.cuda())
        except:
            cuda_output = None
        try:
            cpu_output = self.linear_cpu(x)
        except RuntimeError:
            cpu_output = None
        lin_discrepancy = (cuda_output is not None) and (cpu_output is None)

        # Return boolean indicating any discrepancy (CUDA succeeds where CPU fails)
        return torch.tensor([mat_discrepancy or lin_discrepancy], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(256, 17, dtype=torch.float32)

