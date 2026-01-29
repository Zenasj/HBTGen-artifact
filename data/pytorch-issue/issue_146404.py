# Input is a tuple of two jagged nested tensors (first elements [(2,6), (3,6)], second data (5,6))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        nt0, nt1 = inputs
        nt0_transposed = nt0.transpose(-2, -1)
        return torch.matmul(nt0_transposed, nt1)

def my_model_function():
    return MyModel()

def GetInput():
    nt0 = torch.nested.nested_tensor(
        [torch.rand(2, 6), torch.rand(3, 6)],
        layout=torch.jagged,
        requires_grad=True
    )
    nt1 = torch.nested.nested_tensor_from_jagged(
        torch.rand(5, 6),
        offsets=nt0.offsets()
    ).requires_grad_(True)
    return (nt0, nt1)

