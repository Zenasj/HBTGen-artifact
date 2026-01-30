import torch

import torch.nn as nn

class DesiredModel(nn.Module):
    __constants__ = ['m_names']
    def __init__(self):
        super().__init__()
        m_names = []
        for i in range(3):
            self.add_module(str(i), nn.ReLU())
            m_names.append(str(i))
        self.m_names = m_names

    def forward(self, x):
        o = []
        for m in self._modules.values():
            o.append(m(x))
        return torch.cat(o, 1)

scripted = torch.jit.script(DesiredModel())