import torch
from torch.autograd import gradcheck, gradgradcheck

script = """
def fn(x, y):
    return torch.lerp(x, y, 0.5 + 4j)
"""

torch.jit.CompilationUnit(script)