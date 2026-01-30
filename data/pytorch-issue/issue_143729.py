import torch
import torch.nn as nn

torch.manual_seed(0)
from torch._inductor import config

config.fallback_random = True


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x_frac, x_exp = torch.frexp(x)  # x_frac: int32, x_exp: float32
        x = x_frac * x_exp
        return x


x = torch.randn(4, 1)  # the first element I set 4 can trigger the error
inputs = [x]


def run_test(inputs, mode, device):
    model = Model()

    if device == "cuda":
        model = model.cuda()
        inputs = [x.cuda() for x in inputs]

    if mode == "inductor":
        model = torch.compile(model)

    try:
        output = model(*inputs)
        print(f"{mode} with {device} succeeds: {output}")
    except Exception as e:
        print(f"{mode} with {device} fails: {e}")



run_test(inputs, "eager", "cpu")
run_test(inputs, "inductor", "cpu")  # fail
run_test(inputs, "eager", "cuda")
run_test(inputs, "inductor", "cuda")