# torch.rand(100, 100, dtype=torch.double)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute problematic logdet and correct method (sum(log(svd eigenvalues)))
        logdet_val = torch.logdet(x)
        _, s, _ = torch.svd(x)
        correct_val = torch.sum(torch.log(s))
        return logdet_val - correct_val  # Returns difference between methods

def my_model_function():
    return MyModel()

def GetInput():
    # Generate diagonal matrix with eigenvalues between 1e-4 and 1e-3
    evals = torch.logspace(-4, -3, 100, dtype=torch.double)
    cov = torch.diag(evals)
    return cov

