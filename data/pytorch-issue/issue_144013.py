import torch.nn as nn

import torch


class Model(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv_t = eval(f"torch.nn.ConvTranspose{dim}d(1, 1, kernel_size=(2,) * {dim}, padding=(1,) * {dim})")

    def forward(self, x):
        x = self.conv_t(x)
        x = torch.sigmoid(x)  # tigger condition
        return x


def run_test(dim, mode):
    x = torch.randn(*([1] * (dim + 2)))

    inputs = [x]
    model = Model(dim)

    if mode == "inductor":
        model = torch.compile(model)

    try:
        output = model(*inputs)
        print(f"success on {mode}: {output}")
    except Exception as e:
        print(e)


run_test(1, "eager")
run_test(1, "inductor")

run_test(2, "eager")
run_test(2, "inductor")

run_test(3, "eager")
run_test(3, "inductor")