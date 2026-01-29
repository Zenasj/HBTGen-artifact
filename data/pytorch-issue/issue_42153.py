# torch.rand(1, dtype=torch.float32, device='cuda')  # Dummy input tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Submodules encapsulating the CUDA (faulty) and CPU (correct) implementations
        self.cuda_binomial = torch.distributions.Binomial(
            total_count=torch.tensor(1.0).cuda(),
            probs=torch.tensor(0.9).cuda()
        )
        self.cpu_binomial = torch.distributions.Binomial(
            total_count=torch.tensor(1.0),
            probs=torch.tensor(0.9)
        )
    
    def forward(self, x):
        # Sample large batches on both devices
        cuda_samples = self.cuda_binomial.sample((1000000,))
        cpu_samples = self.cpu_binomial.sample((1000000,)).cuda()  # Move to same device for comparison
        # Check for discrepancies (CUDA might return -1 while CPU doesn't)
        return (cuda_samples != cpu_samples).any()  # Returns True if any differences exist

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input tensor (not used by model's forward)
    return torch.rand(1, device='cuda')

