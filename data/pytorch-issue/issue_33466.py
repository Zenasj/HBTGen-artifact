import torch
import torch.nn as nn

qconv = torch.ops.quantized.conv2d
qconv_prepack = torch.ops.quantized.conv2d_prepack

strides = (1, 1)
pads = (0, 0)
dilations = (1, 1)
groups = 1


for name in ["fbgemm", "qnnpack"]:
    torch.backends.quantized.engine = name
    print("Running on backend ", name)
    x = torch.randn(1, 4, 4, 4)
    qx = torch.quantize_per_tensor(x, scale=0.052, zero_point=0, dtype=torch.quint8)
    weight = torch.randn(2, 4, 2, 2)
    qweight = torch.quantize_per_tensor(weight, scale=2.39, zero_point=0, dtype=torch.qint8)
    w_prepack = qconv_prepack(qweight, None, strides, pads, dilations, groups)
    print(qconv(qx, w_prepack, strides, pads, dilations, groups, 0.112, 0))

tensor([[[[0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000]],

         [[1.2320, 0.2240, 0.0000],
          [0.0000, 0.0000, 2.6880],
          [0.4480, 0.0000, 0.0000]]]], size=(1, 2, 3, 3), dtype=torch.quint8,
       quantization_scheme=torch.per_tensor_affine, scale=0.112, zero_point=0)