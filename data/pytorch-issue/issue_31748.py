import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import check_backward_validity


class RevSequentialBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, rev_block_stack, *inputs):
        check_backward_validity(inputs)

        assert hasattr(rev_block_stack, 'mod_list') and isinstance(rev_block_stack.mod_list, nn.ModuleList)
        assert hasattr(rev_block_stack, 'invert') and callable(rev_block_stack.invert)

        ctx.rev_block_stack = rev_block_stack

        with torch.no_grad():
            outputs = rev_block_stack(*inputs)

        ctx.save_for_backward(*outputs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        bak_outputs = ctx.saved_tensors
        with torch.no_grad():
            for m in ctx.rev_block_stack.mod_list[::-1]:
                inputs = m.invert(*bak_outputs)
                inputs = [t.detach() for t in inputs]
                print(len(inputs))
                for inp in inputs:
                    inp.requires_grad = True

                with torch.enable_grad():
                    outputs = m.forward(*inputs)
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)

                torch.autograd.backward(outputs, grad_output)
                grad_output = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                                    for inp in inputs)
                bak_outputs = inputs

        return (None,) + grad_output


def rev_sequential_backward_wrapper(m, *args):
    return RevSequentialBackwardFunction.apply(m, *args)


class RevChannelPad2D(nn.Module):
    def __init__(self, pad_size, mode='constant', value=0):
        super().__init__()
        assert pad_size >= 0
        self.pad_size = pad_size
        self.mode = mode
        self.value = value

    def forward(self, x):
        # Bug here
        y = F.pad(x, pad=[0, 0, 0, 0, 0, self.pad_size], mode=self.mode, value=self.value)
        # if self.pad_size != 0:
        #     y = F.pad(x, pad=[0, 0, 0, 0, 0, self.pad_size], mode=self.mode, value=self.value)
        # else:
        #     y = x
        return y

    def invert(self, x):
        return x[:, :x.shape[1] - self.pad_size, :, :]


class SimpleRevBlock2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.pad = RevChannelPad2D(out_ch - in_ch)

        self.func = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, 1, 0),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x1, x2):
        x2 = self.pad(x2)
        y = x1 + self.func(x2)
        return x2, y

    def invert(self, y1, y2):
        x2, y = y1, y2
        x1 = y - self.func(x2)
        x2 = self.pad.invert(x2)
        return x1, x2


class RevSequential(nn.Module):
    def __init__(self, modules=None):
        super().__init__()
        self.mod_list = nn.ModuleList(modules)

    def append(self, module):
        assert hasattr(module, 'invert') and callable(module.invert)
        self.mod_list.append(module)

    def extend(self, modules):
        for m in modules:
            self.append(m)

    def forward(self, x1, x2):
        y1, y2 = x1, x2
        for m in self.mod_list:
            y1, y2 = m(y1, y2)
        return y1, y2

    def invert(self, y1, y2):
        x1, x2 = y1, y2
        for m in self.mod_list[::-1]:
            x1, x2 = m.invert(x1, x2)
        return x1, x2

    def __len__(self):
        return len(self.mod_list)

    def __getitem__(self, item):
        return self.mod_list[item]


class Like_IRevNet(nn.Module):
    def __init__(self, use_rev_bw):
        super().__init__()
        self.use_rev_bw = use_rev_bw

        self.seq2 = RevSequential([
            SimpleRevBlock2(32, 32)
        ])

    def forward(self, x):
        x1, x2 = x, x
        y1, y2 = rev_sequential_backward_wrapper(self.seq2, x1, x2)
        y = y1 + y2
        return y


if __name__ == '__main__':
    net = Like_IRevNet(True)
    im = torch.rand(5, 32, 128, 128) + torch.zeros(1, requires_grad=True)
    out = net(im)
    out.sum().backward()