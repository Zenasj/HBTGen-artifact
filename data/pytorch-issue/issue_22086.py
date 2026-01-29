# torch.rand(10, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, k=1, n_samples=10000, replace=False):
        super().__init__()
        self.k = k
        self.n = n_samples
        self.replace = replace

    def forward(self, weights):
        device_cpu = torch.device('cpu')
        device_cuda = torch.device('cuda')

        # Vectorized sampling for CPU
        weights_cpu = weights.to(device_cpu).unsqueeze(0).repeat(self.n, 1)
        samples_cpu = torch.multinomial(weights_cpu, self.k, self.replace)
        all_cpu = samples_cpu.view(-1)

        # Vectorized sampling for CUDA
        weights_cuda = weights.to(device_cuda).unsqueeze(0).repeat(self.n, 1)
        samples_cuda = torch.multinomial(weights_cuda, self.k, self.replace)
        all_cuda = samples_cuda.to(device_cpu).view(-1)

        counts_cpu = torch.bincount(all_cpu, minlength=weights.shape[0])
        counts_cuda = torch.bincount(all_cuda, minlength=weights.shape[0])

        total_samples = self.n * self.k
        p_cpu = counts_cpu / total_samples
        p_cuda = counts_cuda / total_samples

        # Chi-squared test statistic
        chi2_stat = torch.sum((counts_cuda - counts_cpu)**2 / counts_cpu)

        # Critical value for chi2 with 9 degrees of freedom (α=0.05)
        critical_value = 16.919
        return torch.tensor([chi2_stat > critical_value], dtype=torch.bool)

def my_model_function():
    return MyModel(k=1, n_samples=10000, replace=False)

def GetInput():
    return torch.rand(10, dtype=torch.float32)

