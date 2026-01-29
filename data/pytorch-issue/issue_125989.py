import torch
import triton
import triton.language as tl
from torch import nn

# torch.rand(B, C, H, W, dtype=torch.float32, device='cuda') where B=2, C=32, H=8, W=8
@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0.0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

class disable_weight_init:
    class LayerNorm(nn.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__(normalized_shape, eps=eps)
        
        def trition_forward(self, x, weight, bias, eps):
            y = torch.empty_like(x)
            x_arg = x.reshape(-1, x.shape[-1])
            M, N = x_arg.shape
            mean = torch.empty((M,), dtype=torch.float32, device='cuda')
            rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("Feature dimension exceeds Triton kernel limit.")
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            _layer_norm_fwd_fused[(M,)](
                x_arg, y, weight, bias, mean, rstd,
                x_arg.stride(0), N, eps,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1
            )
            return y

        def forward(self, x):
            return self.trition_forward(x, self.weight, self.bias, self.eps)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = disable_weight_init.LayerNorm(32 * 8 * 8)  # C=32, H=8, W=8

    def forward(self, x):
        return self.norm(x.view(x.shape[0], -1)).view(x.shape)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 32, 8, 8, dtype=torch.float32, device='cuda')

