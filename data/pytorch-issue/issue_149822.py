# torch.rand(10, dtype=torch.float32)  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Correct approach (no mixed device outputs)
        correct_values, correct_indices = torch.max(x, 0)
        
        # Incorrect approach (mixed device outputs)
        try:
            if torch.cuda.is_available():
                out_values = torch.empty(10, device="cuda")
                out_indices = torch.empty(10, dtype=torch.long, device="cpu")
                torch.max(x, 0, out=(out_values, out_indices))
                incorrect_values = out_values
                incorrect_indices = out_indices
            else:
                # CPU-only path cannot reproduce the error, return success
                return torch.tensor(True)
            
            # Compare outputs between correct and incorrect approaches
            values_match = torch.allclose(correct_values, incorrect_values)
            indices_match = torch.allclose(correct_indices, incorrect_indices)
            return torch.tensor(values_match and indices_match)
        except RuntimeError:
            # Error indicates the problematic behavior occurred
            return torch.tensor(False)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape of the original code
    return torch.rand(10, dtype=torch.float32)

