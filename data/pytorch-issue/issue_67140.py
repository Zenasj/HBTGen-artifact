# torch.rand(1000, 1, dtype=torch.float64)
import torch
import torch.fft

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # n_full computed as in the original example (n_iter=50)
        self.n_full = 1000 * 50 - 50 + 1

    def forward(self, x):
        # Compute two FFT operations and return their difference
        fft0 = torch.fft.rfft(x, dim=0, n=self.n_full)
        fft1 = torch.fft.rfft(x, dim=0, n=self.n_full)
        return fft0 - fft1  # Difference tensor indicating discrepancy

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input matching the original test case's shape and dtype
    return torch.rand(1000, 1, dtype=torch.float64).cuda()

