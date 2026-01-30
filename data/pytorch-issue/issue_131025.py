import torch
from torch.fx.experimental.proxy_tensor import make_fx


def fn(a, b, c, d):
    x = a + b
    y = c + d
    y.copy_(x)
    x = torch.relu(x)
    x = x.cos().cos()
    return x


a, b, c, d = [torch.randn(2, 4, requires_grad=True) for _ in range(4)]

fx_fn = make_fx(fn)(a, b, c, d)
fx_fn.graph.eliminate_dead_code()
fx_fn.recompile()
print(fx_fn)

def forward(self, a_1, b_1, c_1, d_1):
    add = torch.ops.aten.add.Tensor(a_1, b_1);  a_1 = b_1 = None
    add_1 = torch.ops.aten.add.Tensor(c_1, d_1);  c_1 = d_1 = None
    copy_ = torch.ops.aten.copy_.default(add_1, add);  add_1 = None
    relu = torch.ops.aten.relu.default(add);  add = None
    cos = torch.ops.aten.cos.default(relu);  relu = None
    cos_1 = torch.ops.aten.cos.default(cos);  cos = None
    return cos_1