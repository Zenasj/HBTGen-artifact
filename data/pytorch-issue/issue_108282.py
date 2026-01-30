import torch
import functorch.experimental.control_flow  # for the cond
from functorch import make_fx

def f(x):
   return torch.ops.higher_order.cond(x > 0, torch.sin, torch.cos, [x])

x = torch.tensor(0.5)
gm = make_fx(f, tracing_mode='fake')(x)