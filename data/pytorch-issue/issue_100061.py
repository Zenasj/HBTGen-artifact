import torch.nn as nn

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.switch = True

    def forward(self, X):
        if self.switch:
            return torch.zeros_like(X)
        return torch.ones_like(X)

def test_it(do_compile=False):
    X = torch.arange(5).float()
    model = Model()

    if do_compile:
        model = torch.compile(model)
        print("compiled the model")

    outputs0 = model(X)
    model.switch = False
    outputs1 = model(X)

    torch.testing.assert_close(outputs0, torch.zeros(5))
    torch.testing.assert_close(outputs1, torch.ones(5))

if __name__ == '__main__':
    print(torch.__version__)
    test_it(do_compile=False)  # passes
    print("tests without compile passed")
    test_it(do_compile=True)  # fails
    print("tests with compilie passed")