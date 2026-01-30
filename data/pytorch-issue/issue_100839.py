import torch
import numpy as np

def f(x: torch.Tensor, y: torch.Tensor):
    a, b = x.numpy(), y.numpy()
    c = np.add(x, y)
    return torch.from_numpy(c)

def fn(x, y):
    a = x.numpy()
    b = y.numpy()
    torch._dynamo.graph_break()
    return np.add(a, 1), np.add(b, 1)

def fn(x: np.ndarray, y: np.ndarray):
    a = x.real
    b = y.real
    torch._dynamo.graph_break()
    return np.add(a, 1), np.add(b, 1)