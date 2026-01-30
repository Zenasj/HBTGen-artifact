import torch.nn as nn

import torch
from torch import nn

class subsubmodule(nn.Module):
    def __init__(self, input_size, output_size):
        super(subsubmodule, self).__init__()
        self.emb = nn.Embedding(input_size, output_size)
        
    def forward(self, x):
        pass
    
class submodule(nn.Module):
    def __init__(self, input_size, output_size):
        super(submodule, self).__init__()
        self.emb = subsubmodule(input_size, output_size)
        
    def forward(self, x):
        pass
    
class model(nn.Module):
    def __init__(self, input_size, output_size):
        super(model, self).__init__()
        self.encode = submodule(input_size, output_size)
        self.emb = self.encode.emb.emb  # I want to use a shortcut in some case
        
    def forward(self, x):
        pass
    
if __name__ == "__main__":
    a = model(100, 32)
    print(a)
    print("len of state_dict:", len(a.state_dict()))
    for k,v in a.state_dict().items():
        print(k, v.shape)
    print(a.emb is a.encode.emb.emb)