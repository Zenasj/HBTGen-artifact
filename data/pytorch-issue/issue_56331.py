import torch.nn as nn

py
import torch
import torch.nn.quantized as nnq
### 1 channel
# inpus act
x = torch.Tensor([[[[3,9,30],[14,4,22],[11,7,5]]]])
xq = torch.quantize_per_tensor(x, scale = 1.0, zero_point = 0, dtype=torch.quint8)
xq.int_repr()
c = nnq.Conv2d(1,1,3)
# weights
weight = torch.Tensor([[[[17,21,-59],[-4,-10,-31],[2,-3,59]]]])
qweight = torch.quantize_per_channel(weight, scales=torch.Tensor([1.0]).to(torch.double), zero_points = torch.Tensor([0]).to(torch.int64), axis=0, dtype=torch.qint8)
c.set_weight_bias(qweight,  torch.Tensor([0.0]))
c.scale = 32.0
c.zero_point = 0
out = c(xq)
out

tensor([[[[0.]]]], size=(1, 1, 1, 1), dtype=torch.quint8,
       quantization_scheme=torch.per_tensor_affine, scale=32.0, zero_point=0)