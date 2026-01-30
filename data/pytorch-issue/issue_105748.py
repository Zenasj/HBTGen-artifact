import time
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

dim, n_heads = 128, 1
seqlen = 128
batch_size = 2


model = te.TransformerLayer(
    hidden_size=dim,
    ffn_hidden_size=dim * 4,
    num_attention_heads=n_heads)
model.to("cuda").to(torch.float16)
inp = torch.randn([
    batch_size,
    seqlen,
    dim], device="cuda", dtype=torch.float16, requires_grad=True)

fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.E4M3)
fp8 = True

def forward():
    with te.fp8_autocast(enabled=fp8, fp8_recipe=fp8_recipe):
        with torch.no_grad():
            model(inp)

forward() # <- this is necessary to repro the bug
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    forward()
    # <- Crash happens here

import torch

def fn(x, y):
    return torch.cat([x, y])

fn_opt = torch.compile(fn, dynamic=True)
with torch.profiler.profile(record_shapes=True) as prof:
    for i in range(10):
        x = torch.rand(i*2+4, 8, device='cuda')
        y = torch.rand(i*2+5, 8, device='cuda')
        fn_opt(x, y)

prof.export_chrome_trace("h100_issue.json")