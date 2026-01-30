import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(256, 16)

    def forward(self, x):
        return self.fc(x)


model = Model().eval().to(device="cuda", dtype=torch.bfloat16)
input = torch.zeros(10 * 500_000, 256).to(device="cuda", dtype=torch.bfloat16)

exported_program = torch.export.export(
    model,
    args=(input,),
)
torch._inductor.aoti_compile_and_package(exported_program, inductor_configs={"max_autotune": True})