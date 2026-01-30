import torch.nn as nn

import torch
from torch import nn
from torch.nn.utils import spectral_norm as sn1
from torch.nn.utils.parametrizations import spectral_norm as sn2


def run_with_spectral(spectral):
    discriminator = spectral(nn.Linear(10, 1))
    optim = torch.optim.Adam(discriminator.parameters())
    loss = nn.L1Loss()

    X = torch.ones(4, 10)
    real = torch.ones(4, 1)
    fake = torch.zeros(4, 1)

    optim.zero_grad()
    generated = torch.rand(4, 10)

    loss_real = loss(discriminator(X), real)
    loss_fake = loss(discriminator(generated), fake)

    l = loss_real + loss_fake

    l.backward()
    optim.step()


if __name__ == "__main__":
    run_with_spectral(sn1)
    run_with_spectral(sn2)