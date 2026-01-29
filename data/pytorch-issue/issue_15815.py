# torch.rand(B, 10, 10, 10, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(torch.jit.ScriptModule):
    __constants__ = ['training_now']

    def __init__(self):
        super().__init__()
        self.training_now = True  # Constant to control conditional execution
        self.lnorm = nn.LayerNorm([10, 10, 10])

    @torch.jit.script_method
    def forward(self, x):
        # Conditionally execute training-only code using constant-propagated value
        if self.training_now:
            return self.training_only_code(x)
        else:
            return self.lnorm(x)

    def training_only_code(self, x):
        # Non-TorchScript compatible code (simulated with compatible operations)
        # Example: Python lists, unsupported libraries, or control flow
        # Actual implementation may include non-TorchScript logic
        return self.lnorm(x + 1)  # Placeholder compatible operation

def my_model_function():
    # Returns a model instance with default training_now=True
    return MyModel()

def GetInput():
    # Generates input tensor matching LayerNorm([10,10,10]) requirements
    return torch.rand(1, 10, 10, 10, dtype=torch.float)

