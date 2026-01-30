import torch
import torch.nn as nn
import copy
from optimum.quanto import freeze, quantize, qfloat8_e4m3fn

N_SHAPE = 4096
K_SHAPE = 4096

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(K_SHAPE, N_SHAPE, bias=False)

    def forward(self, inp):
        return self.lin1(inp)

model = MyModel().to(torch.float16)
model = model.eval()

device = "cuda"
seed = 42
batch_size = 10

torch.manual_seed(seed)
device = torch.device("cuda")

model = model.to(device)
model_fp16 = copy.deepcopy(model)

weights = qfloat8_e4m3fn
activations = None

print("------ QUANTIZING")
quantize(model, weights=weights, activations=activations)

print("------ FREEZING")
freeze(model)
print(f"Quantized model (w: {weights}, a: {activations})")

inp = torch.rand(batch_size, K_SHAPE, dtype=torch.float16).to(device)
res = model_fp16(inp)

print("----- quanto model call")
res = model(inp)

print("----- quanto model call")
res = model(inp)

print("----- compiling")
model_quanto_compiled = torch.compile(model)

print("----- running forward")
res = model_quanto_compiled(inp)