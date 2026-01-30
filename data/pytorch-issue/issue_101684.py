import torch

def fn():
    a = torch.ones([4], dtype=torch.float64, device="cuda")
    return (a + 2e50)