import torch
import array
torch.frombuffer(buffer=array.array('i', [1, 2, 3]),
                 dtype=torch.int32,
                 count=1,
                 offset=0,
                 device="cuda",
                 requires_grad=False)