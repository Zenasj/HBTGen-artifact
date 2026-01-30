import torch

def _test(tensor):
    return tensor.to(torch.int16).view(torch.bfloat16)
test = torch.compile(_test)
test(torch.zeros((8, 8), dtype=torch.int8, device="cuda"))