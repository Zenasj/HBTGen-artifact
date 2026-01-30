import torch
import numpy as np


class LinearScheduler(torch.jit.ScriptModule):

    __constants__ = ['i', 'drop_values']
    def __init__(self, module, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.module = module
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps).tolist()

    def forward(self, x):
        return self.module(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.module.drop_prob = self.drop_values[self.i]

        self.i += 1