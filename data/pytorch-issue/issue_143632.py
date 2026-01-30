import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_state: int = 8):
        super().__init__()
        self.embed = nn.Embedding(32, n_state)

    def forward(self, inputs):
        padding = torch.zeros((1, 1), device=inputs.device, dtype=inputs.dtype)
        padded = torch.cat((padding, inputs), dim=0)
        return torch.stack((self.embed(padded), self.embed(padded)))


model = Model().to("cuda")
inputs = torch.randint(0, 32, (1, 1)).to("cuda")

model = torch.compile(model)

with torch.no_grad():
    x1 = model(inputs)
x2 = model(inputs)

print(torch.allclose(x1, x2))

x1, x2