import torch

@torch.jit.script
def test(device: str):
    torch.ones(1,2,3).to(device)

test("cpu")