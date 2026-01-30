import torch

with torch.device('meta'):
     m = BatchNorm(...)
m.load_state_dict(state_dict, assign=True)