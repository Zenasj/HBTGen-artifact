import torch
import torch.nn.functional as F

for x_ in x.split(len(x) // 4):
    with torch.no_grad(): xx = x_.cpu().requires_grad_()
    yy = F.log_softmax(x_.float())
    ...