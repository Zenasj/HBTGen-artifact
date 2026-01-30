import torch.nn as nn

import torch
from torch import nn
from math import log, pi

logabs = lambda x: torch.log(torch.abs(x))

use_weightnorm = True


class ActNorm(nn.Module):
    def __init__(self, in_channel, pretrained=False):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))

        self.initialized = pretrained

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        _, _, T = x.size()

        if not self.initialized:
            self.initialize(x)
            self.initialized = True

        log_abs = logabs(self.scale)
        logdet = torch.mean(log_abs)

        return self.scale * (x + self.loc), logdet


class DebugNet(nn.Module):
    def __init__(self, in_channel, pretrained=False):
        super().__init__()

        self.actnorm = ActNorm(in_channel, pretrained=pretrained)

        self.upsample_conv = nn.ModuleList()
        # 16x upsampling of c
        for s in [16]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            # weight_norm seems to nullify the (totally unrelated) gradient of ActNorm parameters if wrapped in DataParallel
            if use_weightnorm:
                convt = nn.utils.weight_norm(convt)
            self.upsample_conv.append(convt)

    def forward(self, x, c):
        # upsample c by 16x
        c = self.upsample(c)

        # actnorm is not dependent on c at all
        out, logdet = self.actnorm(x)

        # maximum likelihood loss
        log_p = 0.5 * (- log(2.0 * pi) - out.pow(2)).mean()
        return log_p, logdet

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c


use_cuda = True
device = device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 4
# random data (1 for waveform & 40-band spectrogram for c)
random_data = torch.randn(batch_size, 1, 512).clone().detach().to(device)
random_data_c = torch.randn(batch_size, 40, 32).clone().detach().to(device)

net = DebugNet(in_channel=1, pretrained=False).to(device)

# cast the net into DataParallel
net = nn.DataParallel(net)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# train
for i in range(1):
    optimizer.zero_grad()
    log_p, logdet = net(random_data, random_data_c)
    log_p, logdet = torch.mean(log_p), torch.mean(logdet)
    loss = -(log_p + logdet)
    loss.backward()
    optimizer.step()
    # is there grad at actnorm?
    hmm = net.module.actnorm.scale.grad
    if hmm is None:
        print("no grad at actnorm parameters")
    else:
        print("found grad at actnorm parameters")
        print(hmm)