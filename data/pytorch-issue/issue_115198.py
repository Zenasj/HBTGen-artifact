import torch

def g():
    return torch.zeros(1, 2)

exported_module = torch._export.capture_pre_autograd_graph(g, ())