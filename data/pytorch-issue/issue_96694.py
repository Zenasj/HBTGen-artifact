import torch
import torch._inductor.config
torch._inductor.config.debug = True
x = torch.tensor(1, device='cuda')
def forward():
    minus_x = -x # -1 
    minus_x_float = minus_x.to(dtype=torch.float32) # -1.
    return torch.abs(minus_x_float) # 1. ?
print(forward())
fn_compiled = torch.compile(forward)
print(fn_compiled()) # -1. Wrong!