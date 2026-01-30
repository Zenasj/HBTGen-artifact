import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from apex import amp


def BatchNorm2d(out_chan, momentum=0.1, eps=1e-3):
    return nn.SyncBatchNorm.convert_sync_batchnorm(
        nn.BatchNorm2d(out_chan, momentum=momentum, eps=eps)
    )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn = BatchNorm2d(16)
        self.act = nn.ReLU(inplace=True)
        self.linear = nn.Linear(16, 1000)

    def forward(self, x):
        feat = self.act(self.bn(self.conv(x)))
        feat = torch.mean(feat, dim=(2, 3))
        logits = self.linear(feat)
        return logits


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank',
                       dest='local_rank',
                       type=int,
                       default=-1,)
    args = parse.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )

    model = Model()
    criteria = nn.CrossEntropyLoss()
    model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.016,
        weight_decay=1e-5,
        momentum=0.9
    )

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    local_rank = dist.get_rank()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank, ], output_device=local_rank)

    ims = torch.randn(1, 3, 224, 224).cuda()
    lbs = torch.randint(0, 1000, (1, )).cuda()
    logits = model(ims)
    loss = criteria(logits, lbs)
    optimizer.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    print(loss)


if __name__ == '__main__':
    main()