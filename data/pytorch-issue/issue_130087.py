import torch.nn as nn

import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constant tensors (similar to the model parameters such as weights)
        self.y1 = torch.ones(2)
        self.y2 = torch.zeros(2)
        
        self.ref_y1_id = id(self.y1)
        self.ref_y2_id = id(self.y2)
    
    def forward(self, x, ref_id):
        if ref_id == id(self.y1):
            x = torch.mul(x, self.y1)
        else:
            x = torch.mul(x, self.y2)
        return x


x = torch.ones(2)

m1 = M()
compiled_m1 = torch.compile(m1, fullgraph=True)

prediction = compiled_m1(x, m1.ref_y1_id)
prediction = compiled_m1(x, m1.ref_y2_id)