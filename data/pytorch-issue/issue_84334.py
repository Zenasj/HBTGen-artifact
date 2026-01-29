# torch.rand(100, 100, dtype=torch.float32), torch.rand(100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, tolerance=1e-3):
        super(MyModel, self).__init__()
        self.tolerance = tolerance

    def forward(self, inputs):
        a, b = inputs
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            # Compute with TF32 enabled
            torch.backends.cuda.matmul.allow_tf32 = True
            r1 = torch.matmul(a, b)
            # Compute with TF32 disabled
            torch.backends.cuda.matmul.allow_tf32 = False
            r2 = torch.matmul(a, b)
        finally:
            # Restore original TF32 setting
            torch.backends.cuda.matmul.allow_tf32 = original_tf32

        # Return True if difference exceeds tolerance (indicating TF32 effect)
        difference = torch.abs(r1 - r2).sum()
        return difference > self.tolerance

def my_model_function():
    # Returns a model instance with default tolerance
    return MyModel()

def GetInput():
    # Generate two random matrices for matmul testing
    a = torch.rand(100, 100, dtype=torch.float32, device='cuda')
    b = torch.rand(100, 100, dtype=torch.float32, device='cuda')
    return (a, b)

