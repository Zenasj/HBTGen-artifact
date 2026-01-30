import torch
import torch.nn as nn
import torch.nn.functional as F

kl_loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
input = F.log_softmax(torch.randn(3, 5, requires_grad=True))
# Sample a batch of distributions. Usually this would come from the dataset
target = F.softmax(torch.rand(3, 5))
output = kl_loss(input, target)

log_target = F.log_softmax(torch.rand(3, 5))
output = kl_loss(input, log_target, log_target=True)