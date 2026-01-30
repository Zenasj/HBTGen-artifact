import torch.nn as nn

import torch
import tqdm


def net():
    return torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 256)
    )


if __name__ == "__main__":
    pred = net().to('cuda:0')
    opt = torch.optim.Adam(pred.parameters())
    for i in tqdm.trange(100):
        opt.zero_grad()
        pred(torch.randn(2, 1024).to('cuda:0')).sum().backward()
        opt.step()