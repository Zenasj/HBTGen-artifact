import torch.nn as nn

class MyAdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
       

    def forward(self, x): 
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,
                  kernel_size= (inp_size[2], inp_size[3]))