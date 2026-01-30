import torch
from torch import _dynamo as torchdynamo

@torchdynamo.optimize()
def reduce_example(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_out = torch.view_as_complex(torch.stack([x.float(), y.float()], dim=-1))
    return x_out
for _ in range(100):
    reduce_example(torch.randn(1,1,1,128), torch.randn(1,1,1,128))