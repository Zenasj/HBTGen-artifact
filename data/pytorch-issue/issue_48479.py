import torch


def func(t_1: torch.Tensor, t_2: torch.Tensor):
    result = (t_1 + t_2) % 2

    return result


jit_func = torch.jit.script(func)

t_1 = torch.tensor([0], dtype=torch.long, device='cuda:0')
t_2 = torch.tensor([0], dtype=torch.long, device='cuda:0')

jit_func(t_1, t_2)
jit_func(t_1, t_2)