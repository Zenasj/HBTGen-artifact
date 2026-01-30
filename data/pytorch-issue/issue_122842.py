import torch
import torch.nn as nn

# author a model.
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 8)
        self.relu4 = nn.ReLU()
        
    def forward(self, x):
        layer1 = self.relu1(self.fc1(x))
        layer2 = self.relu2(self.fc2(layer1))
        layer3 = self.relu3(self.fc3(layer2))
        y      = self.relu4(self.fc4(layer3))
        return y
    
    def name(self):
        return "MLP"

model = MLPNet().eval()

# generate a uniform distribution of data.
n_batches = 100
x_in = torch.distributions.uniform.Uniform(-1, 1).sample([n_batches, 64, 64])

# export the module (this will annotate the graph with node names such as
# "fc1_weight", "fc1_bias", ...), but not argument names (ex: "x").
m_export = torch.export.export(model, (x_in[0,:],))
print('---------------')
print('torch.export():')
print('---------------')
m_export.module().graph.print_tabular()
print()

import torch
import torch.nn as nn
import torch.export
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config
    )

# author a model.
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
    def forward(self, x, y):
        layer10 = self.relu1(self.fc1(x))
        layer11 = self.relu1(self.fc1(y))
        out = self.relu2(self.fc2(layer10 + layer11))
        return out
    
    def name(self):
        return "MLP"

model = MLPNet().eval()

# generate a uniform distribution of data.
n_batches = 100
# generate some example input.
x_in = torch.distributions.uniform.Uniform(-1, 1).sample([n_batches, 64, 64])

# Quantization.
from torch.export._trace import _export
m_pre = _export(model, (x_in[0,:], x_in[0, :]), pre_dispatch=True).module()
print('-----------------')
print('_trace._export():')
print('-----------------')
print(list(m_pre.graph.nodes))
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())
m_annot = prepare_pt2e(m_pre, quantizer)
for x in x_in:
    m_annot(x, x)
m_quant = convert_pt2e(m_annot, fold_quantize=False)
print('---------------')
print('convert_pt2e():')
print('---------------')
print(list(m_quant.graph.nodes))
m_export = torch.export.export(m_quant, (x_in[0,:], x_in[0,:]))
print('---------------')
print('torch.export():')
print('---------------')
print(list(m_export.module().graph.nodes))