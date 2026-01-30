import torch

if __name__ == "__main__":
    a = torch.rand((36, 48, 48)).requires_grad_()
    a = torch.triu(a)
    a.any()