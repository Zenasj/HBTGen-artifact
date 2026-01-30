import torch

state_dict = torch.load(model_name.pt)
arg = state_dict['args']