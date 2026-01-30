import torch
torch.ones([]).index_select(0, torch.zeros([0], dtype=int))
# Should return 1
# Output can be deterministic, such as tensor(-0.3725)