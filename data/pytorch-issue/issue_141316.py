import torch.nn as nn

import torch
from torch._inductor import config as inductor_config

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.relu(x)
        y = y.to(torch.float8_e4m3fn)
        x = self.sigmoid(x)
        return x, y


inductor_config.aot_inductor.dump_aoti_minifier = True
torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "compile_error"

with torch.no_grad():
    model = Model().to("cuda")
    example_inputs = (torch.randn(8, 10).to("cuda"), torch.randn(8, 10).to("cuda"),)
    ep = torch.export.export(model, example_inputs)
    package_path = torch._inductor.aoti_compile_and_package(ep)
    compiled_model = torch._inductor.aoti_load_package(package_path)
    result = compiled_model(*example_inputs)

# from torch.nn import *
# class Repro(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()



#     def forward(self, y):
#         _to_copy = torch.ops.aten._to_copy.default(y, dtype = torch.float8_e4m3fn);  y = None
#         return (_to_copy,)