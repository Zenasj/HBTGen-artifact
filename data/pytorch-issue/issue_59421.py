# torch.rand(1000, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dropout_cpu = nn.Dropout(p=0.1)
        self.dropout_cuda = nn.Dropout(p=0.1)

    def forward(self, x):
        # Compute CPU path
        x_cpu = x.to('cpu')
        out_cpu = self.dropout_cpu(x_cpu)
        
        # Compute CUDA path (requires CUDA availability)
        if torch.cuda.is_available():
            x_cuda = x.to('cuda')
            out_cuda = self.dropout_cuda(x_cuda)
            out_cuda = out_cuda.to('cpu')  # Move back for comparison
        else:
            # Fallback for no CUDA (returns False discrepancy)
            return torch.tensor(False)
        
        # Compare outputs: True if any element differs (NaN vs 0)
        return torch.any(out_cpu != out_cuda)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.full((1000,), float('inf'), dtype=torch.float32)

