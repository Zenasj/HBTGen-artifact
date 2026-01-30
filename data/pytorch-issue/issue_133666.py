py
import torch
from torch.overrides import TorchFunctionMode, BaseTorchFunctionMode

torch.set_default_device("cuda")

with BaseTorchFunctionMode():
    torch.set_default_device("cpu")
    x = torch.ones(2, 2)
    print(x.device)
    l = []
    for i in range(torch._C._len_torch_function_stack()):
        l.append(torch._C._pop_torch_function_stack())
    
    for m in l:
        torch._C._push_on_torch_function_stack(m)

    print(l)