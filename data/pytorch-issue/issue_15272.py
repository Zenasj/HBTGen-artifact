import copy
import torch
import torch.nn as nn
from torchvision import models


class ReNet(nn.Module):

    def __init__(self, n_input, n_units):
        super(ReNet, self).__init__()

        self.rnn = nn.GRU(n_input, n_units,
                          num_layers=1, batch_first=False,
                          bidirectional=True)

    def rnn_forward(self, x):

        b, n_height, n_width, n_filters = x.size()

        x = x.view(b * n_height, n_width, n_filters)
        x = x.permute(1, 0, 2)
        x, _ = self.rnn(x)
        x = x.permute(1, 0, 2)
        x = x.view(b, n_height, n_width, -1)

        return x

    def forward(self, x):
                                       #b, nf, h, w
        x = x.permute(0, 2, 3, 1)      #b, h, w, nf
        x = self.rnn_forward(x)        #b, h, w, nf
        x = x.permute(0, 3, 1, 2)      #b, nf, h, w

        return x


class Architecture(nn.Module):

    def __init__(self):
        super(Architecture, self).__init__()

        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-5])

        self.renet1 = ReNet(256, 50)

    def forward(self, x):
        x = self.cnn(x)
        x = self.renet1(x)

        return x


def trace(model, usegpu):
    with torch.set_grad_enabled(False):
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        
        if usegpu:
            dummy_input = dummy_input.to(torch.device('cuda'))

        traced_model = torch.jit.trace(model, dummy_input)

    return traced_model


torch.manual_seed(13)

model_cpu = Architecture()
traced_model_cpu = trace(model_cpu, False)
torch.jit.save(traced_model_cpu, "model_cpu.pth")

traced_model_loaded = torch.jit.load("model_cpu.pth")

traced_model_loaded = traced_model_loaded.cuda()
gpu_dummy_input = torch.randn(1, 3, 224, 224).cuda()

traced_model_loaded(gpu_dummy_input)

import torch
from torch import nn

class Model(torch.jit.ScriptModule):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm = torch.jit.trace(torch.nn.LSTM(batch_first=True, 
                                                  input_size=3, 
                                                  hidden_size=3).cuda(),
                                    torch.randn(1,3,3).cuda())
        
    @torch.jit.script_method
    def forward(self, targets):
        return self.lstm(targets)

sm = Model().cpu()                 
sm(torch.tensor([[.1,.2,.3]]))   # no combination of device type will work here