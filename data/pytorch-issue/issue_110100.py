import torch.nn as nn

import torch
from torch._subclasses import fake_tensor

fake_mode = fake_tensor.FakeTensorMode()

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        return out

with fake_mode:
    x = torch.rand(5, 2, 2)
    model = Model()
    # gm, _ = torch._dynamo.export(model, x)  # this works
    exported_program = torch.export.export(model, (x,))  # this fails with AssertionError: fake mode (<torch._subclasses.fake_tensor.FakeTensorMode object at 0x7fe4259ac760>) from active fake mode 0 doesn't match mode (<torch._subclasses.fake_tensor.FakeTensorMode object at 0x7fe355ae0760>) from fake tensor input 0

with maybe_disable_fake_tensor_mode():
        if detected_fake_mode := detect_fake_mode(fake_inps):
            fake_mode = detected_fake_mode