3
import torch
n_features = 0
n_actions = 5
weights = torch.rand(n_actions, n_features)
bias = torch.rand(n_actions)
features = torch.rand(n_features)
logits = weights @ features + bias