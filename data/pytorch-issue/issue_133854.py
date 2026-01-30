import torch
import torch.nn.functional as F

def torch_conv(x, weight):
    _, d_model, L = x.shape
    kernel_size = weight.shape[-1]
    print(f"torch_conv: {x.requires_grad=}, {weight.requires_grad=}")
    y = F.conv1d(
        x,
        weight,
        bias=None,
        stride=1,
        padding=kernel_size - 1,
        groups=d_model,
    )
    print(f"conv out: {y.requires_grad=}")
    y = y[..., :L]
    print(f"conv out sliced: {y.requires_grad=}")
    
    return y

bs = 1
np = 1
hn = 768
seqlen = 8192
dtype = torch.float32
device = "cuda"

x = torch.randn(bs, np * hn, seqlen, dtype=dtype, device=device).requires_grad_()
w = torch.randn(d, 1, hl, dtype=dtype, device=device).requires_grad_()
out = torch_conv(x, w)