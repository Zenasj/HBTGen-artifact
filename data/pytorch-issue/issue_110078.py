import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 20)
        self.eps = 1e-12
        self.dim = -1
        
    def forward(self, x):
        denom = x.norm(2.0, self.dim, keepdim=True)
        print("denom", denom.dtype)
        return denom

model = MyModel()
model = model.eval().to(torch.float16).to("cuda")

inp = torch.rand(8, 10, device="cuda", dtype=torch.float16)

with torch.no_grad():
    res = model(inp)

    torch.onnx.export(model, (inp,), f="normalize.onnx")