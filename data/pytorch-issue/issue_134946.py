import torch.nn as nn

import torch
from torch._inductor import config as inductor_config
inductor_config.cpp_wrapper = True
inductor_config.freezing = True
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(512, 512).bfloat16()

    def forward(self, x):
        return self.linear1(x)

model = Model().eval()
inputs = torch.randn(128, 512)
with torch.no_grad(), torch.amp.autocast("cpu", enabled=True, dtype=torch.bfloat16):
    so_path = torch._export.aot_compile(
         model, (inputs, ),
         options={"aot_inductor.output_path":os.path.join(os.getcwd(), "model.so")}
    )