import torch.nn as nn

3
import torch


model = torch.nn.Linear(10, 20).cuda()
x = torch.randn(1, 10).cuda()

with torch.autocast(device_type="cuda", enabled=True): # when switch to False, it runs well
    with torch.no_grad():
        _ = model(x)
    
    out = model(x)
    loss = out.mean()
    loss.backward()

3
import torch


model = torch.nn.ModuleList(
    [torch.nn.LayerNorm(10), torch.nn.Linear(10, 20), torch.nn.LayerNorm(20)]
).cuda()
x = torch.randn(1, 10).cuda()

with torch.autocast(device_type="cuda", enabled=True):
    with torch.no_grad():
        o = x
        for layer in model:
            o = layer(o)
    out = x
    for layer in model:
        out = layer(out)
    loss = out.mean()
    loss.backward()
    for layer in model:
        print(f"{layer=}, {layer.weight.grad=}")