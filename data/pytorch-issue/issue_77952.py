import torch.nn as nn

import torch
input_tensor = torch.tensor([[0.2332, 0.1621, 0.2437, 0.1001]], dtype=torch.float32)
target_tensor = torch.tensor([[5, 5, 6, 2]], dtype=torch.int64)

reduction = "mean"

input = input_tensor.clone().detach().to('cuda')
target = target_tensor.clone().detach().to('cuda')
res1 = torch.nn.functional.multilabel_margin_loss(input, target, reduction=reduction, )

input = input_tensor.clone().detach().to('cuda')
target = target_tensor.clone().detach().to('cuda')
res2 = torch.nn.functional.multilabel_margin_loss(input, target, reduction=reduction, )

input = input_tensor.clone().detach().to('cuda')
target = target_tensor.clone().detach().to('cuda')
res3 = torch.nn.functional.multilabel_margin_loss(input, target, reduction=reduction, )

print(res1) # tensor(3.3126, device='cuda:0')
print(res2) # tensor(1.0626, device='cuda:0')
print(res3) # tensor(1.0626, device='cuda:0')