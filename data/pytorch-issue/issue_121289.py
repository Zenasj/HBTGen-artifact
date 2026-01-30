import torch 
import torch_mlu  
torch.triu_indices(3, 3, device='mlu')