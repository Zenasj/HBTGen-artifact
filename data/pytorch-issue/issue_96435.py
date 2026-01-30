import torch
import torch._dynamo

def fn(x, scale, zero_point):
    return torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

x = torch.rand((2, 2), requires_grad=True) * 2 + 10
scale = torch.tensor(2.)
zero_point = torch.tensor(10.)

torch._dynamo.optimize("aot_eager")(fn)(x, scale, zero_point)