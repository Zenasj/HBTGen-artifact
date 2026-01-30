import torch
import torch.nn.attention.flex_attention


class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(256, 256 * 3)
        self.n_head = 256 // 64
        self.d_attn = 256

    def forward(self, x):
        n_batch, n_ctx, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.d_attn, self.d_attn, self.d_attn], dim=2)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1)
        k = k.reshape(n_batch, n_ctx, self.n_head, -1)
        v = v.reshape(n_batch, n_ctx, self.n_head, -1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = torch.nn.attention.flex_attention.flex_attention(q, k, v)
        return x


torch.set_default_device("cuda")
torch.manual_seed(0)

repro = Repro()
x = torch.randn((1, 512, 256))
out = torch.compile(repro, backend="aot_eager")(x)
out.sum().backward()

import torch
import torch.nn as nn

class Repro(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(256, 256 * 3)
        self.n_head = 256 // 64
        self.d_attn = 256

    def forward(self, x):
        n_batch, n_ctx, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.d_attn, self.d_attn, self.d_attn], dim=2)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        k = k.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        v = v.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        return nn.functional.scaled_dot_product_attention(q, k, v)

torch.set_default_device("cuda")
torch.manual_seed(0)

model = Repro()
compiled_model = torch.compile(Repro())

x = torch.randn((1, 512, 256), requires_grad=True)
x_compiled = x.clone().detach().requires_grad_(True)

out = model(x)
out_compiled = compiled_model(x_compiled)
out.sum().backward()
out_compiled.sum().backward()

weight_diff = torch.max(torch.abs(model.qkv_proj.weight.grad - compiled_model.qkv_proj.weight.grad)).item()
bias_diff = torch.max(torch.abs(model.qkv_proj.bias.grad - compiled_model.qkv_proj.bias.grad)).item()

print(f"Weight grad max abs diff: {weight_diff:.2e}")
print(f"Bias grad max abs diff: {bias_diff:.2e}")