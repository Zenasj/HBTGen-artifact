import torch

@torch.jit.script
def fn(x):
    # type: (List[int]) -> List[int]
    new_list = [0]
    for i in x:
        if i == 2:
            x.remove(i)
        new_list.append(i)
    return new_list