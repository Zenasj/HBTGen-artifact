import torch.nn as nn

import torch.nn.functional as F

input = torch.randn(1, 3, 224, 224).cuda()

weight_original = torch.randn(16, 3, 3, 3).cuda()
weight_original.requires_grad = True
weight_tea = torch.randn(16, 3, 3, 3).cuda()
weight_tea.requires_grad = True

out1_o = F.conv2d(input, weight_original)
out1_t = F.conv2d(input, weight_tea)
loss1 = (out1_o - out1_t).mean()
loss1.backward(retain_graph=True, create_graph=True)

weight_tea.grad.zero_()

weight_original = weight_original + weight_original.grad

out2 = F.conv2d(input, weight_original)
loss2 = out2.mean()
loss2.backward()
print(weight_tea.grad)