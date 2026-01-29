# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Attempt to trigger CUDA allocator behavior with an extremely large allocation
        # This mimics the test case from the issue's C++ extension
        try:
            _ = torch.empty(1 << 60, device='cuda')  # 1 EB allocation (exceeds typical GPU capacity)
        except RuntimeError as e:
            # The PR changes the error message here (from internal assert to OOM)
            pass
        return x + 1  # Dummy computation to ensure the module is valid

def my_model_function():
    # Returns the model instance
    return MyModel()

def GetInput():
    # Returns a minimal valid input tensor (unused in problematic allocation logic)
    return torch.rand(1, dtype=torch.float32)

