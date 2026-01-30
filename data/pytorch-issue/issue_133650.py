import torch
from torch._dynamo.comptime import comptime

@torch._dynamo.config.patch(do_not_emit_runtime_asserts=True, capture_scalar_outputs=True)
@torch.compile(dynamic=True, fullgraph=True, backend="eager")
def cf_printlocals(x):
    u5, u3 = x[2:].tolist()
    u6, *u10 = x.tolist()
    u4 = x[1].item()
    u9, u8, *u11 = x[:-1].tolist()
    torch._check(u3 != 1)
    torch._check(u5 != u6 + 2 * u4)
    torch._check_is_size(u6)
    torch._check_is_size(u4)
    torch._check_is_size(u5)
    torch._check((u6 + 2*u4) % u5 == 0)
    torch._check(u3 == (u6 + 2 * u4) // u5)
    comptime.print({
        "u5": u5,
        "u3": u3,
        "u6": u6,
        "u10": u10,
        "u4": u4,
        "u9": u9,
        "u8": u8,
        "u11": u11,
    })
    u2 = torch.randn(u5, u3)
    u0 = torch.zeros(u6)
    torch._check_is_size(u4)
    u1 = torch.zeros(u4 * 2)
    stk = torch.cat([u0, u1], dim=0)
    return torch.stack([stk, stk]).view(2, *u2.size())

cf_printlocals(torch.tensor([20, 2, 3, 8]))