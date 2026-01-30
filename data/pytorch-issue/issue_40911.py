import torch.nn as nn
import random

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import SGD
import numpy as np
import torch


def dummy_dl(batch_size, patch_size, data_channels, num_classes):
    data = np.random.random((batch_size, data_channels, *patch_size))
    seg = np.random.uniform(0, num_classes-1, (batch_size, 1, *patch_size)).round()
    data_torch = torch.from_numpy(data).float()
    seg_torch = torch.from_numpy(seg).float()
    while True:
        yield data_torch, seg_torch


def main():
    patch_size = [320, 256]
    batch_size = 40
    data_channels = 1
    num_classes = 2

    model = Generic_UNet(data_channels, 16, num_classes, 6, 2, 2, nn.Conv2d, nn.InstanceNorm2d,
                         {'eps': 1e-05, 'affine': True}, nn.Dropout2d,
                         {'p': 0, 'inplace': True}, nn.LeakyReLU, {'negative_slope': 0.01, 'inplace': True}, False,
                         False, lambda x: x, InitWeights_He(1e-2), [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
                         [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], False, True, True, 512)
    model = model.cuda()
    optimizer = SGD(model.parameters(), 0.01, 0.99, nesterov=True, weight_decay=3e-5)

    dl = dummy_dl(batch_size, patch_size, data_channels, num_classes)
    dl_val = dummy_dl(batch_size, patch_size, data_channels, num_classes)

    loss_fn = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    scaler = GradScaler()
    for ep in range(1000):
        print(ep)
        losses_train = []
        for batch in range(30):
            optimizer.zero_grad()

            data, seg = next(dl)
            data = data.cuda()
            seg = seg.cuda()

            with autocast():
                output = model(data)
                loss = loss_fn(output, seg)

            scaler.scale(loss).backward()
            #loss.backward()

            scaler.step(optimizer)
            #optimizer.step()

            scaler.update()

            losses_train.append(loss.item())
        print(np.mean(losses_train))

        losses_val = []
        with torch.no_grad():
            for batch in range(30):
                data, seg = next(dl_val)
                data = data.cuda()
                seg = seg.cuda()

                with autocast(True):
                    output = model(data)
                    loss = loss_fn(output, seg)
                losses_val.append(loss.item())
        print(np.mean(losses_val))


if __name__ == "__main__":
    main()