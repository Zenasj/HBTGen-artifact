import torch.nn as nn

import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, dtype):
        super(SimpleModel, self).__init__()
        self.gemm1 = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        return out

model = SimpleModel(torch.float).to(torch.float8_e4m3fn)

torch.save(model, "save_model_file.pt")
loaded_model = torch.load("save_model_file.pt")
print(f"loaded_model = {loaded_model}")

torch.save(model.state_dict(), "model_state_dict.pt")
model.load_state_dict(torch.load("model_state_dict.pt"))
print(f"model with state dict loaded = {model}")

example_inputs = torch.randn([4, 4]).to(torch.float8_e4m3fn)
exported_model = torch.export.export(model, (example_inputs,))
print(f"exported_model = {exported_model}")
torch.export.save(exported_model, "exported_model.pt")

import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, dtype):
        super(SimpleModel, self).__init__()
        self.gemm1 = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        out = self.gemm1(x)
        out = self.relu1(out)
        return out

model = SimpleModel(torch.float).to(torch.float8_e4m3fn)


example_inputs = torch.randn([4, 4]).to(torch.float8_e4m3fn)
exported_model = torch.export.export(model, (example_inputs,))
print(f"exported_model = {exported_model}")
torch.export.save(exported_model, "exported_model.pt")