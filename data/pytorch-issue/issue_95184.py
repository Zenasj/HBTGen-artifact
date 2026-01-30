import torch.nn as nn

import torch

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, scalar):
        tensor = tensor.__or__(scalar)
        return tensor

if __name__ == "__main__":
    tensor = torch.tensor([7, 2, 3, 4], dtype=torch.uint8)
    model = BasicModule()
    model = torch.compile(model)
    fwd_res = model(tensor, 7)
    print(fwd_res)