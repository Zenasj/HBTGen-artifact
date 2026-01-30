import torch

z = torch.pow(x, y)

aten.pow.Scalar,
aten.pow.Tensor_Scalar,
aten.pow.Tensor_Tensor,

torch.pow(x, y, out=z)

aten.pow.Scalar_out,
aten.pow.Tensor_Scalar_out,
aten.pow.Tensor_Tensor_out,

x.pow_(y)

aten.pow_.Scalar,
aten.pow_.Tensor,