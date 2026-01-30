import torch.nn as nn

m = random_pruning(nn.Linear(5, 7), name='weight', amount=0.2)
prune.random_pruning(m, name='weight', amount=0.2)