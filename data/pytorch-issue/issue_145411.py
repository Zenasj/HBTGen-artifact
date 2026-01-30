import torch.nn as nn

import copy
import logging

import torch

from torch.nn import functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x

def main():
    with torch.no_grad():
        model = Model()
        example_inputs = (
            torch.randn(8, 10),
            torch.randn(8, 10),
        )
        ep = torch.export.export(model, example_inputs)
        package_path = torch._inductor.aoti_compile_and_package(ep)
        compiled_model = torch._inductor.aoti_load_package(package_path)
        copy.deepcopy(compiled_model) # this line errors with TypeError: cannot pickle 'torch._C._aoti.AOTIModelPackageLoader' object



if __name__ == "__main__":
    main()