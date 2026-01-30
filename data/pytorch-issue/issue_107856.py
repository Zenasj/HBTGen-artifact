import torch
import torch._dynamo

device = torch.device("cpu")

def my_func(target):
    target_device = target.device
    assert isinstance(target_device, torch.device)
    a = torch.zeros(2, 3, device=target_device)
    b = torch.zeros(2, 3, device=target_device)
    c = torch.zeros(2, 3, device=target_device)
    return a + b + c

opt_func = torch._dynamo.optimize("inductor")(my_func)

a = torch.tensor([2, 3], device=device)
res = opt_func(a)