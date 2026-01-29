# torch.rand(2, 4, 3, dtype=torch.float32)  # Inferred input shape (batch, vectors, dim)
import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def forward(self, x):
        xx = F.normalize(x, p=2, dim=-1)
        cov = torch.stack([W.T @ W for W in xx])  # Compute covariance matrices
        
        # Standard path (ensures contiguous tensor)
        cov_standard = cov.sum(0, keepdim=True).contiguous()
        logdet_standard = torch.logdet(torch.eye(3, device=x.device) + 0.1 * cov_standard).sum()
        
        # Faulty path (simulates non-contiguous tensor after reduction)
        cov_faulty = cov.sum(0, keepdim=True)
        cov_faulty = cov_faulty.permute(0, 2, 1).permute(0, 2, 1)  # Force non-contiguous view
        logdet_faulty = torch.logdet(torch.eye(3, device=x.device) + 0.1 * cov_faulty).sum()
        
        # Return difference between valid and faulty paths' outputs
        return logdet_standard - logdet_faulty

def my_model_function():
    return MyModel()

def GetInput():
    # Generates a random input matching the expected shape and requirements
    return torch.randn(2, 4, 3, dtype=torch.float32, requires_grad=True)

