# torch.rand(3, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute matrix exponential on CPU
        x_cpu = x.to('cpu')
        cpu_out = torch.matrix_exp(x_cpu)
        
        # Compute matrix exponential on GPU if available
        if torch.cuda.is_available():
            x_gpu = x.to('cuda')
            gpu_out = torch.matrix_exp(x_gpu).to('cpu')  # Move back to CPU for comparison
            # Return comparison result as a boolean tensor
            return torch.tensor(torch.allclose(cpu_out, gpu_out, equal_nan=True), dtype=torch.bool)
        else:
            # If no GPU, assume outputs are identical (return True)
            return torch.tensor(True, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Replicate the input tensor from the issue with inf, nan, and 5.0
    return torch.tensor(
        [[float('inf'), float('nan'), 5.0],
         [float('inf'), float('nan'), 5.0],
         [float('inf'), float('nan'), 5.0]],
        dtype=torch.float32
    )

