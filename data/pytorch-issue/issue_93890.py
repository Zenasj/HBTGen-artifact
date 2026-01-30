import torch

torch.autocast

def fn(x):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = torch.mul(x, 5)
        torch._dynamo.graph_break()
        x = torch.relu(x)
    return x

x = torch.rand([4, 4])
print(fn(x))
opt_fn = torch._dynamo.optimize("inductor")(fn)
print(opt_fn(x))