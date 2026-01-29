# (torch.rand(3, dtype=torch.bfloat16), torch.rand(3, dtype=torch.bfloat16))
import torch
import torch._refs as refs

class MyModel(torch.nn.Module):
    def forward(self, x):
        a, b = x
        out_torch = torch.fmod(a, b)
        out_ref = refs.fmod(a, b)
        # Compare outputs, treating NaNs as not equal (as in the issue's failing case)
        are_close = torch.allclose(out_torch, out_ref, atol=1e-5, equal_nan=False)
        return torch.tensor([are_close], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(3, dtype=torch.bfloat16)
    b = torch.rand(3, dtype=torch.bfloat16) * 1e-37 + 1e-38  # Generate small values similar to test case
    return (a, b)

