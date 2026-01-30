import torch.nn as nn

prior = torch.ones(32145, 4)
torch.save(prior,  'prior.pth')

import torch
from torch import nn

class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key,value in tensor_dict.items():
            setattr(self, key, value)

prior = torch.ones(32145, 4)
tensor_dict = {'prior': prior}
tensors = TensorContainer(tensor_dict)
tensors = torch.jit.script(tensors)
tensors.save('prior.pth') # I believe this is the torch.jit.save() thing that your error message is asking for.