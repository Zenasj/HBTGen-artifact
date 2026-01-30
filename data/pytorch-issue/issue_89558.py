import torch.nn as nn

import torch
torch.autograd.set_detect_anomaly(True)
input_data = torch.tensor([[1.5, 1.3]], requires_grad=True) # This data produces [-0.6, -0.8] log probability
#input_data = torch.tensor([[0.01, -700]], requires_grad=True)  # This data produces [0, -700] log probability
target_prob = torch.tensor([[1.0, 0.0]], requires_grad=True)
input_logprob = torch.nn.functional.log_softmax(input_data, dim=1)
print(input_logprob)
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
output = kl_loss(input_logprob, target_prob)
print(output)
output.backward()