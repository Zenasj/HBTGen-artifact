import torch

tensor = torch.rand(torch.Size([]))
res1 = torch.movedim(tensor, 0, 0)
# RuntimeError: std::distance(source_dims.begin(), source_iter) == rest_dimINTERNAL ASSERT FAILED at "../aten/src/ATen/native/TensorShape.cpp":2448, please report a bug to PyTorch.

import torch
import numpy

t = torch.randn([])
numpy.moveaxis(t.tolist(), 0, 0)
# AxisError: source: axis 0 is out of bounds for array of dimension 0

import torch

func_cls=torch.Tensor.movedim

tensor = torch.rand(torch.Size([]))
def test():
    tmp_result= func_cls(tensor, 0, 0)
    return tmp_result
res1 = test()