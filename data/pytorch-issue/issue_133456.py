# torch.rand(3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Correct approach using a single combined boolean mask
        correct_x = x.clone()
        combined_mask = torch.tensor([True, False, False], device=x.device)
        correct_x[combined_mask] = 2.0

        # Incorrect approach using chained boolean indexing (creates a copy)
        incorrect_x = x.clone()
        mask1 = torch.tensor([True, True, False], device=x.device)
        temp = incorrect_x[mask1]  # View of first two elements
        temp[torch.tensor([True, False], device=x.device)] = 2.0  # Assignment affects a copy, not original

        return incorrect_x, correct_x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

