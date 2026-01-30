import torch.nn as nn

import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.linear = torch.nn.Linear(in_feat, out_feat)

    def forward(self, x):
        out = self.linear(x)
        return out

from setuptools import setup
from Cython.Build import cythonize

setup(name='test', ext_modules=cythonize('simple_model.py'))

from simple_model import SimpleModel
import torch
from torch.export import export

def main():

    model = SimpleModel(2, 4)
    example_inputs = (torch.rand(4,2),)
    export(model, example_inputs)

if __name__ == '__main__':
    main()