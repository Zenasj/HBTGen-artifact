import torch

def onnx_cdists(x1, x2, p=2):
    """ Custom cdists function for ONNX export since neither cdists nor
    linalg.norm is currently support by the current PyTorch version 1.10.
    """
    s_x1 = torch.unsqueeze(x1, dim=1)
    s_x2 = torch.unsqueeze(x2, dim=0)
    diffs = s_x1 - s_x2
    if p == 1:
        return diffs.abs().sum(dim=2)
    elif p == 'inf':
        return diffs.abs().max(dim=2).values
    elif p % 2 == 0:
        return diffs.pow(p).sum(dim=2).pow(1/p)
    else:
        return diffs.abs().pow(p).sum(dim=2).pow(1/p)