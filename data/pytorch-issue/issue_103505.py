import torch.nn as nn

import torch
from torch.nn import Linear
from torchdistx.fake import fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch._functorch.aot_autograd import aot_export_joint_simple


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(1024, 4096)
        self.linear2 = Linear(4096, 1024)

    def forward(self, x):
        y = self.linear(x)
        z = self.linear2(y)
        # loss = torch.sum(z)
        return tuple([z])


with FakeTensorMode():
    sample_input = torch.randn(4, 512, 1024)
    loss = torch.rand(4, 512, 4096)

    model = TestModel()
    z = model(sample_input)

    graph_module = aot_export_joint_simple(model, tuple([sample_input]), trace_joint=True)
    print(graph_module)

from transformers import BertModel, BertConfig
with torch.device("meta"):
    sample_input = torch.randint(0, 30522, [4, 512])

    model = BertModel(BertConfig())
    z = model(sample_input)

    graph_module = aot_export_module(model, tuple([sample_input]),output_loss_index=0, trace_joint=True)
    print(graph_module)

import torch
from torch.nn import Linear
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor

class TransformerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tf = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(1024, 16, 4096), 24)
    
    def forward(self, x):
        y = self.tf(x)
        loss = torch.sum(y)
        return (loss,)

with FakeTensorMode():
    model = TransformerModel()
    sample_input = torch.randn(4, 512, 1024)

    graph_module, sig = aot_export_module(model, tuple([sample_input]), trace_joint=True, output_loss_index=0)