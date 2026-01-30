import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import functional as F

class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.p = torch.tensor(4).cuda() 

    def forward(self, x):
        size_array = [int(s) for s in x.size()[2:]]
        x = F.avg_pool2d(x.pow(self.p), size_array, stride=1).pow(1.0 / self.p)
        return x

def export_onnx(modelfile, onnxfile):
    model = AvgPool()
    model.cuda()
    model.eval()
    dummy_input = Variable(torch.randn(1, 3, 800, 900)).cuda()

    dynamic_axes = {'input': {2 : 'batch1', 3 : 'batch2'}}
    torch.onnx.export(model, dummy_input, onnxfile, verbose=True, \
                      aten=False, \
                      opset_version=10, \
                      training=False, \
                      do_constant_folding=True, \
                      keep_initializers_as_inputs=True, \
                      input_names=('input', ), \
                      output_names=('output',), \
                      dynamic_axes=dynamic_axes)

import torch
import torch.nn as nn
from torch.nn import functional as F

@torch.jit.script
def avg_pool_helper(x, p):
    size_array = [int(s) for s in x.size()[2:]]
    x = F.avg_pool2d(x.pow(p), size_array, stride=1).pow(1.0 / p)
    return x

class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.p = torch.tensor(4).cuda() 

    def forward(self, x):
        return avg_pool_helper(x, self.p)