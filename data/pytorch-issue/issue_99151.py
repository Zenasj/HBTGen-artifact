import torch.nn as nn

import torch
arg_1_tensor = torch.randint(2, 3, [5, 5], dtype=torch.float32)
arg_2_tensor = torch.randint(2, 3, [5, 5], dtype=torch.float32)
arg_3 = None
arg_4 = "mean"
res = torch.nn.functional.binary_cross_entropy(input=arg_1_tensor,target=arg_2_tensor,weight=arg_3,reduction=arg_4,)
print(res)
# res: RuntimeError: all elements of input should be between 0 and 1

import torch
arg_1_tensor = torch.randint(2, 4, [10, 64], dtype=torch.float32)
arg_2_tensor = torch.randint(2, 4, [10, 64], dtype=torch.float32)
arg_3 = None
arg_4_tensor = torch.rand([64], dtype=torch.float32)
arg_4 = arg_4_tensor.clone()
arg_5 = "mean"
res = torch.nn.functional.binary_cross_entropy_with_logits(input=arg_1_tensor,target=arg_2_tensor,weight=arg_3,pos_weight=arg_4,reduction=arg_5,)
print(res)
# res: tensor(-3.8518)