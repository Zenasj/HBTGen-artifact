# torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, A):
        # Compute on CPU
        A_cpu = A.to('cpu')
        eival_cpu, eivec_cpu = torch.linalg.eigh(A_cpu)
        # Sort eigenvalues and eigenvectors
        sorted_indices_cpu = torch.argsort(eival_cpu)
        eival_cpu_sorted = eival_cpu[sorted_indices_cpu]
        eivec_cpu_sorted = eivec_cpu[:, sorted_indices_cpu]
        # Align eigenvector signs by first element's sign
        signs_cpu = torch.sign(eivec_cpu_sorted[0, :])
        eivec_cpu_sorted = eivec_cpu_sorted * signs_cpu

        # Compute on GPU
        A_gpu = A.to('cuda')
        eival_gpu, eivec_gpu = torch.linalg.eigh(A_gpu)
        sorted_indices_gpu = torch.argsort(eival_gpu)
        eival_gpu_sorted = eival_gpu[sorted_indices_gpu]
        eivec_gpu_sorted = eivec_gpu[:, sorted_indices_gpu]
        signs_gpu = torch.sign(eivec_gpu_sorted[0, :])
        eivec_gpu_sorted = eivec_gpu_sorted * signs_gpu

        # Compare eigenvalues and eigenvectors
        eigenvals_close = torch.allclose(eival_cpu_sorted, eival_gpu_sorted.cpu(), atol=1e-6)
        eigenvecs_close = torch.allclose(eivec_cpu_sorted, eivec_gpu_sorted.cpu(), atol=1e-6)
        return torch.tensor(eigenvals_close and eigenvecs_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.rand(3, 3, dtype=torch.float32)
    A = (A + A.T) / 2  # Ensure symmetric matrix
    return A

