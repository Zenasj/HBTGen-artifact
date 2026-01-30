import torch.nn as nn

import torch
class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_dict):
        return torch.nn.functional.embedding_bag(**input_dict)


if __name__ == "__main__":
    input_dict = {
        "input": torch.LongTensor(
            [3, 1, 1, 5, 4, 3, 2, 6, 3, 5, 6, 5, 6, 0, 6, 4, 3, 5, 7, 7]
        ),
        "weight": torch.tensor(
            [
                [0.4597, 0.3764],
                [0.0013, 0.1590],
                [0.3964, 0.0136],
                [0.2254, 0.2905],
                [0.8277, 0.8274],
                [0.6265, 0.7479],
                [0.1626, 0.6431],
                [0.1259, 0.1363],
            ],
            requires_grad=True,
        ),
        "offsets": torch.LongTensor([0, 3, 10]),
        "mode": "mean",
        "sparse": False,
        "scale_grad_by_freq": False,
        "max_norm": -1.0,
    }

    model = BasicModule()
    model = torch.compile(model)
    fwd_res = model(input_dict)
    print(fwd_res)

import torch

@torch.compile(backend="aot_eager")
def f(x):
    x.detach().add_(1)
    return x * 2

out = f(torch.ones(2, requires_grad=True))

x.mul_(2).detach().mul_(3)

def f(x):
    x.mul_(2)
    with torch.no_grad():
        x.mul_(3)
    x.mul_(4)