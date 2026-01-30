import torch
import torch._dynamo
import logging

# torch._logging.set_logs(dynamo=logging.DEBUG, aot=logging.DEBUG, inductor=logging.DEBUG)

def gn(x, y):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z = torch.mm(x, y)
        torch._dynamo.graph_break()
        return torch.sin(z)


def fn(x, y):
    z = torch.mm(x, y)
    z = z + gn(x, y)
    return z


x = torch.rand(3, 3, device="cuda")
y = torch.rand(3, 3, device="cuda")

print(fn(x, y))
opt_fn = torch.compile(backend="eager")(fn)
print(opt_fn(x, y))

tensor([[1.3335, 1.5669, 1.3017],
        [0.6343, 2.1670, 1.0982],
        [0.6622, 1.2455, 1.1671]], device='cuda:0')
tensor([[1.3359, 1.5703, 1.2969],
        [0.6328, 2.1719, 1.0938],
        [0.6641, 1.2500, 1.1719]], device='cuda:0', dtype=torch.bfloat16)