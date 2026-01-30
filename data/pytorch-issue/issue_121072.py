import torch
from torch.nn.functional import interpolate

def get_grads(dtype, size):
    input = torch.randn((2, 2, 1, 1), dtype=dtype, requires_grad=True).cuda()

    output = interpolate(input, size=size, mode='bicubic', align_corners=True)

    grad_outputs = [torch.ones_like(output)]
    grads = torch.autograd.grad([output], [input], grad_outputs)
    return grads[0]

print(get_grads(torch.float32, (128, 128)).flatten())
print(get_grads(torch.float16, (128, 128)).flatten())
print(get_grads(torch.bfloat16, (128, 128)).flatten())

print(get_grads(torch.float32, (64, 64)).flatten())
print(get_grads(torch.float16, (64, 64)).flatten())
print(get_grads(torch.bfloat16, (64, 64)).flatten())

print(get_grads(torch.float32, (32, 32)).flatten())
print(get_grads(torch.float16, (32, 32)).flatten())
print(get_grads(torch.bfloat16, (32, 32)).flatten())

tensor([16384., 16384., 16384., 16384.], device='cuda:0')
tensor([2048., 2048., 2048., 2048.], device='cuda:0', dtype=torch.float16)
tensor([256., 256., 256., 256.], device='cuda:0', dtype=torch.bfloat16)
tensor([4096., 4096., 4096., 4096.], device='cuda:0')
tensor([2048., 2048., 2048., 2048.], device='cuda:0', dtype=torch.float16)
tensor([256., 256., 256., 256.], device='cuda:0', dtype=torch.bfloat16)
tensor([1024., 1024., 1024., 1024.], device='cuda:0')
tensor([1024., 1024., 1024., 1024.], device='cuda:0', dtype=torch.float16)
tensor([256., 256., 256., 256.], device='cuda:0', dtype=torch.bfloat16)

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Conv2d(3, 3, 3, padding=1, bias=False)

    def forward(self, x):
        print("1 x dtype", x.dtype)
        x = self.m(x)
        print("2 x dtype", x.dtype)
        x = nn.functional.interpolate(x, scale_factor=(2.5, 2.5), mode="bicubic", align_corners=False)
        print("3 x dtype", x.dtype)
        return x

def func(enabled, dtype=torch.float16):
    torch.manual_seed(12)
    model = Model().cuda()
    x = torch.rand(2, 3, 32, 32, device="cuda")

    with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
        out = model(x)
        print(out.dtype)
        loss = out.mean()

    loss.backward()
    grad = model.m.weight.grad
    print(grad.dtype, grad.max(), grad.min(), grad.mean())


print("\n--- No AMP")
func(False)
print("\n--- AMP using f16")
func(True, dtype=torch.float16)
print("\n--- AMP using bf16")
func(True, dtype=torch.bfloat16)