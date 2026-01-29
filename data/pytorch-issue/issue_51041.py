# torch.rand(B, dtype=torch.int64)  # Input is a 1D tensor of integers
import torch
from torch import nn
from typing import Optional, List, Dict

class MyModel(nn.Module):
    # Correctly annotated attributes to avoid TorchScript type inference issues
    foo: Optional[List[int]] = None  # For List[int] handling
    bar: Optional[Dict[int, str]] = None  # For Dict[int, str] handling

    def __init__(self):
        super().__init__()
        self.foo = None  # Initialize to None (valid due to class-level annotation)
        self.bar = {}  # Empty dict is allowed with class-level annotation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input tensor to Python list and dict for attribute assignment
        x_list = x.tolist()
        x_dict = {i: str(v) for i, v in enumerate(x_list)}  # Example dict creation

        # Assign to attributes (validated by TorchScript due to proper annotations)
        self.foo = x_list
        self.bar = x_dict

        # Return a dummy output (matches original issue's return type semantics)
        return torch.tensor(1.0)

def my_model_function() -> nn.Module:
    return MyModel()

def GetInput() -> torch.Tensor:
    # Generate a random 1D tensor of integers (shape (5,))
    return torch.randint(0, 10, (5,), dtype=torch.int64)

