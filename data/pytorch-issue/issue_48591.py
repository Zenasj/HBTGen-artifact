# torch.rand(1, 2, 2, 8, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Create contiguous copy of input
        x_contig = x.contiguous()
        
        # Apply in-place logit_() to both original and contiguous tensors
        x.logit_()
        x_contig.logit_()
        
        # Compute logit with 1e-6 threshold on both modified tensors
        x_logit = x.logit(1e-6)
        x_contig_logit = x_contig.logit(1e-6)
        
        # Compare using tolerances from the test failure message
        equal = torch.allclose(
            x_logit, x_contig_logit,
            rtol=1.3e-6,  # from error message "rtol=1.3e-06"
            atol=1e-5     # from error message "atol=1e-05"
        )
        return torch.tensor(equal, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Create channels_last tensor as in the issue's reproduction code
    return torch.randn((1, 2, 2, 8), dtype=torch.float32).contiguous(memory_format=torch.channels_last)

