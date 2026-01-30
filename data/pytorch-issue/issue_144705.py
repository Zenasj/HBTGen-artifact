import torch
from torch._higher_order_ops.out_dtype import out_dtype


def quantized_matmul(x_vals_int8, x_scales, w_vals_int8):
    return out_dtype(torch.ops.aten.mm.default, torch.int32, x_vals_int8, w_vals_int8) * x_scales


x_vals_int8 = torch.randn(65536, 144).to(dtype=torch.int8).cuda()
x_scales = torch.randn(65536, 1).to(dtype=torch.float32).cuda()
w_vals_int8 = torch.randn(432, 144).to(dtype=torch.int8).cuda().t()

qcm = torch.compile(quantized_matmul, mode='max-autotune-no-cudagraphs')

qcm(x_vals_int8, x_scales, w_vals_int8)