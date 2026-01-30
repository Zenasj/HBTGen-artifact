import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.q = nn.Parameter(torch.rand(1, 64, 64, 96, dtype=torch.float16).to("cuda"))
        self.k = nn.Parameter(torch.rand(1, 64, 64, 96, dtype=torch.float16).to("cuda"))
        self.v = nn.Parameter(torch.rand(1, 64, 64, 96, dtype=torch.float16).to("cuda"))

    def forward(self, x):
        return torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=True, dropout_p=0.0) * x

model = MyModel()

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    inp = torch.Tensor([3.]).to("cuda", torch.float16)
    res = model(inp)

    loss = torch.sum(res)

    loss.backward()

import torch

t = torch.nn.Transformer(d_model=576).cuda()
x = torch.randn(1, 10, 576).cuda()
with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    t(x, x).mean().backward()

import torch

print("python.__version =", torch.__version__)
print("device =", torch.cuda.get_device_name())

d_models=[384, 512, 560, 576, 640, 1024, 1400]

for d_model in d_models:
    try:
        print('d_model =', d_model, end='')
        t = torch.nn.Transformer(d_model=d_model).cuda()
        x = torch.randn(1, 10, d_model).cuda()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            t(x, x).mean().backward()
        print(" -> OK")
    except RuntimeError as e:
        print(" -> " + str(e))