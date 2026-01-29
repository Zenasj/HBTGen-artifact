# torch.rand((), dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpu_submodel = nn.Identity()  # Submodule on CPU
        self.cuda_submodel = nn.Identity()  # Submodule on CUDA (if available)
        # Initialize weights to ensure outputs differ when device mismatch occurs
        # (Identity modules have no learnable parameters, so this is just a placeholder)
        self.cpu_submodel.weight = nn.Parameter(torch.tensor([1.0]))
        self.cuda_submodel.weight = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        # Process input on both CPU and CUDA devices
        cpu_out = self.cpu_submodel(x.to('cpu'))
        cuda_out = self.cuda_submodel(x.to('cuda'))
        
        # Compare device types and values (moved to same device)
        device_mismatch = (cpu_out.device != cuda_out.device)
        value_close = torch.allclose(cpu_out, cuda_out.to(cpu_out.device))
        
        # Return a tensor indicating comparison results:
        # [device_mismatch (1 if mismatch), value_close (1 if close)]
        return torch.tensor([int(device_mismatch), int(value_close)], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random 0-dimensional tensor on CPU (base device)
    return torch.rand((), dtype=torch.float32)

