import torch.nn as nn
import numpy as np

class LocalResponseNorm(Module):
    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.) -> None:
        ...

import torch

torch.manual_seed(420)

inp = torch.randn(8, 3, 32, 32)

def func(inp):
    torch.onnx.export(torch.nn.LocalResponseNorm(inp.shape[1], 1, 2, 1),inp,"lrn.onnx",input_names=["inp"],output_names=["out"],keep_initializers_as_inputs=True,)
    return inp

func(inp)
# RuntimeError: 0 INTERNAL ASSERT FAILED at 
# "/opt/conda/conda-bld/pytorch_1672906354936/work/torch/csrc/jit/ir/alias_analysis.cpp":608, 
# please report a bug to PyTorch. 
# We don't have an op for aten::add but it isn't a special case. 
# Argument types: Tensor, bool, int,