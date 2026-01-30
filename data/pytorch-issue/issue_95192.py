import torch.nn as nn

import torch
class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return torch.is_floating(input=tensor)

if __name__ == "__main__":
    tensor = torch.randn([1024], dtype=torch.float32)
    model = BasicModule()
    model = torch.compile(model)
    fwd_res = model(tensor)
    print(fwd_res)