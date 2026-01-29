# torch.rand(1024 ** 3, dtype=torch.float32, pin_memory=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.copy_stream = torch.cuda.Stream()  # CUDA stream for data transfer

    def forward(self, A):
        with torch.cuda.stream(self.copy_stream):
            B = torch.zeros(1024 ** 3, device='cuda:0')
        torch.cuda.synchronize()  # Wait for B creation to complete
        
        # Simulate long-running computation on B in default stream
        for _ in range(100):
            B.mul_(2)
        del B  # Memory freed here
        
        # Attempt to reuse CUDA stream for data transfer (may reuse B's memory)
        with torch.cuda.stream(self.copy_stream):
            C = A.to('cuda:0', non_blocking=True)
        
        torch.cuda.synchronize()  # Ensure all operations complete
        return C

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1024 ** 3, dtype=torch.float32, pin_memory=True)

