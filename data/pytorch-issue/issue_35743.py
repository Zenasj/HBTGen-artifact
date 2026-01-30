import torch.nn as nn

class custom_AdaptiveMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        inp_size = x.size()
        size1=inp_size[2]
        size2=inp_size[3]
        return nn.functional.max_pool2d(input=x,
                  kernel_size= (size1, size2))

class custom_AdaptiveMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,
                  kernel_size= (inp_size[3], inp_size[3]))

import torch.onnx
torch_out = torch.onnx._export(network,             # model being run
                               specs,                       # model input (or a tuple for multiple inputs)
                               "sample.onnx")