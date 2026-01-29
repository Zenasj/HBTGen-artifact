# torch.rand(100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        x_sym = x + x.T  # Enforce symmetry as in the example
        # Call torch.lobpcg with k=1 (inferred from error context) and largest=False
        eigenvalues, _ = torch.lobpcg(x_sym, k=1, largest=False)
        return eigenvalues  # Return eigenvalues to trigger backward computation

def my_model_function():
    # Returns the model instance with default parameters
    return MyModel()

def GetInput():
    # Returns a random symmetric input tensor (enforced via x + x.T in the model)
    return torch.rand(100, 100, dtype=torch.float32)

