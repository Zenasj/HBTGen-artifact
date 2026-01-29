# torch.rand(2, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        n = int(1e7)  # Replicate original issue's tensor size
        self.z = torch.randn(2, n, dtype=torch.float32, device="cpu")  # Fixed large tensor as part of the model
        
    def forward(self, x):
        batch_sz = int(1e7 / 10)  # Original batch size calculation
        batch_lst = []
        for i in range(0, self.z.size(1), batch_sz):
            batch = self.z[:, i:i+batch_sz].cuda()  # Move batch to GPU
            product = x.view(2, 1).cuda() * batch  # Critical operation causing memory retention
            squared = product ** 2  # Gradient computation retains GPU references
            max_val, _ = torch.max(squared, dim=0)  # Compute max along first dimension
            batch_lst.append(max_val.cpu())  # Move results back to CPU
        f = torch.mean(torch.cat(batch_lst))  # Final aggregation
        return f

def my_model_function():
    return MyModel()  # Returns model instance with problematic forward pass

def GetInput():
    return torch.rand(2, dtype=torch.float32, requires_grad=True)  # Matches model's input requirements

