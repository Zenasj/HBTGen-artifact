def create_yolov3_modules(anchors, anch_mask, n_classes, ignore_thre):
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
    mlist.append(resblock(ch=64))
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
    mlist.append(resblock(ch=128, nblocks=2))
    ...

def add_conv(in_ch, out_ch, ksize, stride):
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage

class resblock(ScriptModule):

    __constants__ = ["nblocks","ch","shortcut"]

    def __init__(self, ch, nblocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.nblocks = nblocks
        self.ch = ch
        self.module_list = nn.ModuleList()
        self.blockt1=add_conv(self.ch, self.ch//2, 1, 1)
        self.blockt2=add_conv(self.ch//2, self.ch, 3, 1)
        for _ in range(nblocks):
            resblock_one = nn.ModuleList()
            self.blockt1
            self.blockt2
            self.module_list.append(resblock_one)

    @script_method
    def forward(self, x):
        for _ in range(self.nblocks):#in_ch, out_ch, ksize, stride
            h = x
            h = self.blockt1(h)
            h = self.blockt2(h)
            x = x + h if self.shortcut else h
        return x

import torch
import torch.nn as nn

def add_conv(in_ch, out_ch, ksize, stride):
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage

class resblock(torch.jit.ScriptModule):

    __constants__ = ["nblocks","ch","shortcut", "blockt1", "blockt2"]

    def __init__(self, ch, nblocks=1, shortcut=True):
        super(resblock, self).__init__()
        self.shortcut = shortcut
        self.nblocks = nblocks
        self.ch = ch
        self.module_list = nn.ModuleList()
        self.blockt1=add_conv(self.ch, self.ch//2, 1, 1)
        self.blockt2=add_conv(self.ch//2, self.ch, 3, 1)
        for _ in range(nblocks):
            resblock_one = nn.ModuleList()
            self.blockt1
            self.blockt2
            self.module_list.append(resblock_one)

    @torch.jit.script_method
    def forward(self, x):
        for _ in range(self.nblocks):#in_ch, out_ch, ksize, stride
            h = x
            h = self.blockt1(h)
            h = self.blockt2(h)
            x = x + h if self.shortcut else h
        return x

mlist = nn.ModuleList()
mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
mlist.append(resblock(ch=64))
mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
mlist.append(resblock(ch=128, nblocks=2))
print(mlist)
r = resblock(2)
print(r.graph)

@torch.jit.script
def inner(x):
    # type: (float) -> float
    return x

@torch.jit.script
def use_float(x):
    # type: (int) -> float
    return inner(x)