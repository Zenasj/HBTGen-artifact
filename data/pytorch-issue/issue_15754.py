import torch


@torch.jit.script
def test(tensor):
    return tensor.cpu()