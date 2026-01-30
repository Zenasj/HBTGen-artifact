import torch.nn as nn

import torch 
class TestKwMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(3, 16)
        self.layer2 = torch.nn.Linear(3, 32)
    def forward(self, x1, x2, flag=True):
        x1o = self.layer1(x1)
        x2o = self.layer2(x2)
        return torch.cat([x1o, x2o], dim=1)
def main():
    mod = TestKwMod()
    gm = torch.export.export(mod, (torch.rand(1, 3), ), {
            "flag": False,
            "x2": torch.rand(1, 3),
        }, strict=False)
    print(gm)
if __name__ == "__main__":
    main()