import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.ops.aten.mul(x1, True)

def main():
    mod = MyModule()
    f = torch.jit.script(mod)
    print(f.graph)

if __name__ == "__main__":
    main()