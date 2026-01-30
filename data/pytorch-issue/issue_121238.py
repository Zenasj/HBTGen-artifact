import torch
dtype = torch.float16

def intermediate(step_t, exp_inf):
    tmp = 0.9 ** step_t
    tmp -= 1 
    return exp_inf * tmp

def forward1(step_t, exp_inf, param, exp_avg):
    denom = -intermediate(step_t, exp_inf)
    div = torch.div(exp_avg, denom)
    mul_1 = torch.mul(div, -1)
    param.add_(mul_1)

def forward2(step_t, exp_inf, param, exp_avg):
    denom = intermediate(step_t, exp_inf)
    param.addcdiv_(exp_avg, denom)
    
step_t = torch.tensor(2, dtype=dtype, device="cuda")
param1 = torch.rand(2, 3, dtype=dtype, device="cuda") 
param2 = param1.clone()
exp_inf = torch.rand(2, 3, dtype=dtype, device="cuda")
exp_avg = torch.rand(2, 3, dtype=dtype, device="cuda")

forward1(step_t, exp_inf, param1, exp_avg)
forward2(step_t, exp_inf, param2, exp_avg)

assert torch.allclose(param1, param2), f"results are not the same! {param1=} {param2=}"