import torch.nn as nn

import torch
arg_1 = torch.as_tensor([2])
arg_2 = torch.rand([2, 3], dtype=torch.float32)
arg_3 = torch.as_tensor([0])
res = torch.nn.functional.embedding_bag(input=arg_1,weight=arg_2,offsets=arg_3)
print(res)