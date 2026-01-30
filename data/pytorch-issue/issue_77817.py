import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
action = 0
advantage = torch.tensor(1.)

# no invalid action masking
print("=============regular=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True)
target_probs = Categorical(logits=target_logits)
print("probs:   ", target_probs.probs)
log_prob = target_probs.log_prob(torch.tensor(action))
print("log_prob:", log_prob)
(log_prob*advantage).backward()
print("gradient:", target_logits.grad)

device = "mps"
action = 0
advantage = torch.tensor(1.)

# no invalid action masking
print("=============regular=============")
target_logits = torch.tensor([1., 1., 1., 1.,] , requires_grad=True, device=device)
target_probs = Categorical(logits=target_logits)
print("probs:   ", target_probs.probs)
log_prob = target_probs.log_prob(torch.tensor(action))
print("log_prob:", log_prob)
(log_prob*advantage).backward()
print("gradient:", target_logits.grad)