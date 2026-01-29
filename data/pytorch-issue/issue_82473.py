# torch.rand(3, 3, dtype=torch.float64, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Run SVD twice and check for discrepancies
        first_svd = torch.linalg.svd(x)
        second_svd = torch.linalg.svd(x)
        
        # Compare U, S, Vh with tolerance (epsilon=1e-6)
        u_close = torch.allclose(first_svd.U, second_svd.U, atol=1e-6)
        s_close = torch.allclose(first_svd.S, second_svd.S, atol=1e-6)
        vh_close = torch.allclose(first_svd.Vh, second_svd.Vh, atol=1e-6)
        
        # Return True if any component differs between first and second call
        return torch.tensor([not (u_close and s_close and vh_close)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Replicate the problematic input tensor from the issue
    input_data = torch.tensor(
        [[56.6896, 4.7862, 10.0108],
         [4.7059, 90.4238, 10.5659],
         [10.3995, 11.0197, 12.6870]],
        dtype=torch.float64).cuda()
    return input_data

