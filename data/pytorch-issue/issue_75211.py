import torch

def reference_global_norm(tensors: List[Tensor]) -> Tensor:
    return torch.sqrt(sum(t.pow(2).sum() for t in tensors))