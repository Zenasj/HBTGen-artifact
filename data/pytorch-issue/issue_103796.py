import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        self.layers : nn.ModuleList = nn.ModuleList()

    def foo(self):
        for _ in self.layers[:-1]:
            pass