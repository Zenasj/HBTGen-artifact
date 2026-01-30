import torch.nn as nn

import torch


class Dummy(torch.nn.Module):
    def forward(self, x):
        y0 = x * 0
        y1 = x * 1
        y0.data = y1  # which breaks jit.trace
        return y0


if __name__ == '__main__':
    inputs = torch.tensor([1])
    model = Dummy()
    model.eval()  # makes no difference, in case someone asks
    outputs = model(inputs)  # get tensor([1])

    # jit
    model_jit = torch.jit.trace(model, (inputs,))
    outputs_jit = model_jit(inputs)  # get tensor([0]), expect tensor([1]).
    print(outputs.equal(outputs_jit))  # False