import torch
import torch.nn as nn
import torch.nn.attention.flex_attention

class Repro(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(256, 256 * 3)
        self.n_head = 256 // 64
        self.d_attn = 256
        self.qkv_proj.weight.data.fill_(0.1)
        self.qkv_proj.bias.data.fill_(0.1)

    def forward(self, x):
        n_batch, n_ctx, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.d_attn, self.d_attn, self.d_attn], dim=2)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        k = k.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        v = v.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        return torch.nn.attention.flex_attention.flex_attention(q, k, v)

torch.set_default_device("cuda")
torch.manual_seed(0)

model = Repro()

compiled_model = Repro()
compiled_model = torch.compile(compiled_model)

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

import functools
import torch
import torch.nn.attention.flex_attention

torch.set_default_device("cuda")

def merge_attn(x_local, lse_local, x_global, lse_global):
    max_lse = torch.maximum(lse_local, lse_global).detach()
    exp_local = torch.exp(lse_local - max_lse)
    exp_global = torch.exp(lse_global - max_lse)
    numerator = (x_local * exp_local[..., None]) + (x_global * exp_global[..., None])
    denominator = exp_local[..., None] + exp_global[..., None]
    merged_x = numerator / denominator
    merged_lse = max_lse + torch.log(exp_local + exp_global)
    return merged_x, merged_lse

def create_masks(n_local_band):
    def sliding_window_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= n_local_band)

    def global_causal_v1(b, h, q_idx, kv_idx):
        return q_idx > kv_idx

    def global_causal_v2(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx > n_local_band)

    sliding_window_causal_mask = torch.nn.attention.flex_attention.create_block_mask(
        sliding_window_causal, B=None, H=None, Q_LEN=512, KV_LEN=512
    )
    global_causal_mask_v1 = torch.nn.attention.flex_attention.create_block_mask(
        global_causal_v1, B=None, H=None, Q_LEN=512 - n_local_band, KV_LEN=512
    )
    global_causal_mask_v2 = torch.nn.attention.flex_attention.create_block_mask(
        global_causal_v2, B=None, H=None, Q_LEN=512, KV_LEN=512
    )

    return sliding_window_causal_mask, global_causal_mask_v1, global_causal_mask_v2

def attn_v1(query, key, value, sliding_window_causal_mask, global_causal_mask):
    n_batch, n_ctx, d_model = query.shape
    n_head, n_local_band = 16, 128

    query = query.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    key = key.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    value = value.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)

    local_attn = functools.partial(torch.nn.attention.flex_attention.flex_attention, block_mask=sliding_window_causal_mask, return_lse=True)
    global_attn = functools.partial(torch.nn.attention.flex_attention.flex_attention, block_mask=global_causal_mask, return_lse=True)

    x_local, lse_local = local_attn(query, key, value)
    x_global, lse_global = global_attn(query[:, :, n_local_band:, :], key, value)

    x, lse = merge_attn(
        x_local[:, :, n_local_band:, :], lse_local[:, :, n_local_band:], x_global, lse_global
    )
    x = torch.concat([x_local[:, :, :n_local_band, :], x], dim=2)
    x = x.transpose(1, 2).contiguous().reshape(n_batch, n_ctx, d_model)
    return x

def attn_v2(query, key, value, sliding_window_causal_mask, global_causal_mask):
    n_batch, n_ctx, d_model = query.shape
    n_head = 16

    query = query.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    key = key.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)
    value = value.reshape(n_batch, n_ctx, n_head, -1).transpose(1, 2)

    local_attn = functools.partial(torch.nn.attention.flex_attention.flex_attention, block_mask=sliding_window_causal_mask, return_lse=True)
    global_attn = functools.partial(torch.nn.attention.flex_attention.flex_attention, block_mask=global_causal_mask, return_lse=True)

    x_local, lse_local = local_attn(query, key, value)
    x_global, lse_global = global_attn(query, key, value)

    x, lse = merge_attn(x_local, lse_local, x_global, lse_global)
    x = x.transpose(1, 2).contiguous().reshape(n_batch, n_ctx, d_model)
    return x

def run_comparison(compile=False):
    n_local_band = 128
    sliding_window_causal_mask, global_causal_mask_v1, global_causal_mask_v2 = create_masks(n_local_band)

    if compile:
        attn_v1_func = torch.compile(attn_v1)
        attn_v2_func = torch.compile(attn_v2)
    else:
        attn_v1_func = attn_v1
        attn_v2_func = attn_v2

    query_v1 = torch.randn(2, 512, 512, requires_grad=True)
    key_v1 = torch.randn(2, 512, 512, requires_grad=True)
    value_v1 = torch.randn(2, 512, 512, requires_grad=True)

    query_v2 = query_v1.clone().detach().requires_grad_(True)
    key_v2 = key_v1.clone().detach().requires_grad_(True)
    value_v2 = value_v1.clone().detach().requires_grad_(True)

    out_v1 = attn_v1_func(query_v1, key_v1, value_v1, sliding_window_causal_mask, global_causal_mask_v1)
    out_v1.sum().backward()

    out_v2 = attn_v2_func(query_v2, key_v2, value_v2, sliding_window_causal_mask, global_causal_mask_v2)
    out_v2.sum().backward()

    print(f"Output difference - Min: {(out_v1 - out_v2).min():.2e}, Max: {(out_v1 - out_v2).max():.2e}")
    
    for name, grad_1, grad_2 in zip(["query", "key", "value"], [query_v1, key_v1, value_v1], [query_v2, key_v2, value_v2]):
        print(f"{name} gradient difference - Min: {(grad_1.grad - grad_2.grad).min():.2e}, Max: {(grad_1.grad - grad_2.grad).max():.2e}")
        print(f"{name} gradients close: {torch.allclose(grad_1.grad, grad_2.grad)}")

print("Without compile:")
run_comparison(compile=False)

print("\nWith compile:")
run_comparison(compile=True)

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
)

shift = 32

def causal_mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001 ARG001 ANN201
    return q_idx >= kv_idx


def causal_mask_slidewindow_mod(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx <= kv_idx + shift)


class Repro(nn.Module):
    def __init__(self, option):  # option in [mask1, mask2, mask1+mask2]
        super().__init__()
        self.qkv_proj = nn.Linear(256, 256 * 3)
        self.n_head = 256 // 64
        self.d_attn = 256
        self.qkv_proj.weight.data.fill_(0.1)
        self.qkv_proj.bias.data.fill_(0.1)
        self.option = option
        self.mask1 = create_block_mask(
                causal_mask_mod,             1, None, 512, 512, _compile=False
            )
        self.mask2 = create_block_mask(
                causal_mask_slidewindow_mod, 1, None, 512, 512, _compile=False
            )

    def forward(self, x):
        n_batch, n_ctx, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.d_attn, self.d_attn, self.d_attn], dim=2)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        k = k.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        v = v.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)

        if   self.option == 'mask1':
            return torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=self.mask1) 
        elif self.option == 'mask2':
            return torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=self.mask2)
        else:
            output = torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=self.mask1) + \
                    torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=self.mask2)
            return output


torch.set_default_device("cuda")
torch.manual_seed(0)

for option in ['mask1', 'mask2', 'mask1+mask2']:

    model = Repro(option=option)

    compiled_model = Repro(option=option)
    compiled_model = torch.compile(compiled_model)

    x = torch.randn((1, 512, 256), requires_grad=True)
    x_compiled = x.clone().detach().requires_grad_(True)

    out = model(x)
    out_compiled = compiled_model(x_compiled)

    out.sum().backward()
    out_compiled.sum().backward()

    weight_diff = torch.max(torch.abs(model.qkv_proj.weight.grad - compiled_model.qkv_proj.weight.grad)).item()
    bias_diff = torch.max(torch.abs(model.qkv_proj.bias.grad - compiled_model.qkv_proj.bias.grad)).item()
    print(f'------------option {option}----------------')
    print(f"Weight grad max abs diff: {weight_diff:.2e}")
    print(f"Bias grad max abs diff: {bias_diff:.2e}")