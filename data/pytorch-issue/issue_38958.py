import torch
import torch.nn as nn
import torch.cuda.amp as amp


class Model(nn.Module):
    def forward(self):
        a = torch.randn(1)
        b = torch.randn(1)
        c = torch.cat((a, b), 0)

        return c


print(torch.__version__)

model = Model()
model_jit_script = torch.jit.script(model)

with amp.autocast(False):
    model()
    model_jit_script()

with amp.autocast(True):
    model()
    print("Relevant variant:", flush=True)
    model_jit_script()