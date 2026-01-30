import torch.nn as nn

import torch

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, param):
        z = torch.frexp(**param)
        return z


if __name__ == "__main__":
    model = Repro()
    params = {"input": torch.randn([128, 256])}
    params["out"] = [
        torch.empty(0, dtype=torch.float32),  # mantissa
        torch.empty(0, dtype=torch.int32),  # exponent
    ]
    model = torch.compile(model, backend="eager")
    res = model(params)
    print(res)