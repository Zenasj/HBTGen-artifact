import torch

def copy_as_numpy(x):
  return t.cpu().numpy() 
torch.Tensor.copy_as_numpy = copy_as_numpy