import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flag = False

    def forward(self, x, flag):
        self.flag = flag.item()
        return torch.cond(
            self.flag,
            lambda x: x * 0.0,
            lambda x: 2 * x,
            (x,),
        )

input1 = (
    torch.randn(1).cuda(),
    torch.tensor([0], dtype=torch.bool),
)
model = M().cuda()
_ = model(*input1)