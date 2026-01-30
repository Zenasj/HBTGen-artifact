def forward(self, x):
    # x: [batch, seq_len, dim]
    seq_length = x.shape[1]
    for i in range(seq_length):
         nn.Linear(xx, yy)

def forward(self, x):
    # x: [batch, seq_len, dim]
    y = torch.tensor(1, dtype = x.dtype)

import torch 
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.linear = nn.Linear(128, 10)
    def forward(self, x):
        return self.linear(x)
    
if __name__ == "__main__":
    model = SimpleModel()
    x = torch.randn(4, 128)
    exported_model = torch.export.export(model, (x,))
    print(exported_model)

import torch 
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.linear = nn.Linear(128, 10)
    def forward(self, x):
        return self.linear(x)
    
if __name__ == "__main__":
    model = SimpleModel()
    x = torch.randn(4, 128)
    exported_model = torch.export.export(model, (x,))
    gm = torch.fx.symbolic_trace(exported_model.module())
    gm.to_folder('xxx', 'yyy')