import torch.nn as nn

import torch
import os

class Model(torch.nn.Module):
    def __init__(self, inp_size, out_size, inner_size):
        super(Model, self).__init__()
        self.Linear1 = torch.nn.Linear(inp_size, inner_size)
        self.Linear2 = torch.nn.Linear(inner_size, out_size)
        self.x = None

    def forward(self, inp):
        res = self.Linear1(inp)
        self.x = res
        return self.Linear2(res)

N, D_in, H, D_out, inner = 2, 2, 2, 2, 4
model = Model(D_in, H, inner).to("cuda")
input = torch.ones(N, D_in, device='cuda')
compiled_model = torch.compile(model, mode="reduce-overhead")


input2 = torch.ones(N, D_in, device='cuda').add_(10)
# Keep some intermediate value in model.x
model.x = torch.Tensor([[100, 100, 100, 100],[200, 200, 200, 200]])

# Expected behavior
out = model(input2)
print(f"Expected model.x = {model.x}")

# Keep some intermediate value in compiled_model.x
compiled_model.x = torch.Tensor([[100, 100, 100, 100],[200, 200, 200, 200]])

# the first run warms up each graph, which does things like CuBlas or Triton benchmarking
out = compiled_model(input2)
print(f"Incorrect compiled_model.x = {compiled_model.x}")

# The second run does a CUDA Graph recording, and replays it
out = compiled_model(input2)
print(f"Incorrect compiled_model.x = {compiled_model.x}")

# Finally we hit the optimized, CUDA Graph replay path
out = compiled_model(input2)
print(f"Incorrect compiled_model.x = {compiled_model.x}")