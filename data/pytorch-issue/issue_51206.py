import torch.nn as nn

from time import time

from torch import nn
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional
from torch.optim import SGD


class Network(nn.Module):
    def __init__(self, cast_for_upsample=False):
        super().__init__()
        self.cast_for_upsample = cast_for_upsample

        self.layers = nn.Sequential(
            nn.Conv3d(1, 32, 3, 1, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),
            nn.Conv3d(32, 32, 3, 1, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(32, 64, 3, 2, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),
            nn.Conv3d(64, 64, 3, 1, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(64, 128, 3, 2, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),
            nn.Conv3d(128, 128, 3, 1, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(128, 256, 3, 2, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),
            nn.Conv3d(256, 256, 3, 1, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),

            nn.Conv3d(256, 512, 3, 2, 1, 1, 1, False),
            nn.LeakyReLU(1e-2, True),
            nn.Conv3d(512, 512, 3, 1, 1, 1, 1, False),
        )

    def forward(self, x):
        down = self.layers(x)
        if self.cast_for_upsample:
            up = nn.functional.interpolate(down.float(), x.shape[2:], None, 'trilinear').half()
        else:
            up = nn.functional.interpolate(down, x.shape[2:], None, 'trilinear')
        return up


if __name__ == "__main__":
    inp = torch.rand((2, 1, 64, 64, 64)).cuda()

    net = Network(cast_for_upsample=False).cuda()
    optimizer = SGD(net.parameters(), 0.001)

    torch.cuda.empty_cache()

    # warmup
    for _ in range(10):
        optimizer.zero_grad()
        out = net(inp)
        l = torch.square(inp - out).mean() # just the MSE between input and output as a dummy loss function
        l.backward()
        optimizer.step()

    # fp32 measurement
    st = time()
    for _ in range(100):
        optimizer.zero_grad()
        out = net(inp)
        l = torch.square(inp - out).mean() # just the MSE between input and output as a dummy loss function
        l.backward()
        optimizer.step()
    print('fp32:', time() - st)

    ####################################################
    # now AMP
    net = Network(cast_for_upsample=False).cuda()
    optimizer = SGD(net.parameters(), 0.001)
    scaler = GradScaler()

    torch.cuda.empty_cache()

    # warmup
    for _ in range(10):
        optimizer.zero_grad()

        with autocast():
            out = net(inp)
            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function

        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()

    # amp measurement
    st = time()
    for _ in range(100):
        optimizer.zero_grad()

        with autocast():
            out = net(inp)
            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function

        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
    print('amp:', time() - st)

    ####################################################
    # now AMP with hacking interpolate so that is runs in fp32
    net = Network(cast_for_upsample=True).cuda()
    optimizer = SGD(net.parameters(), 0.001)
    scaler = GradScaler()

    torch.cuda.empty_cache()

    # warmup
    for _ in range(10):
        optimizer.zero_grad()

        with autocast():
            out = net(inp)
            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function

        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()

    # amp measurement
    st = time()
    for _ in range(100):
        optimizer.zero_grad()

        with autocast():
            out = net(inp)
            l = torch.square(inp - out).mean()  # just the MSE between input and output as a dummy loss function

        scaler.scale(l).backward()
        scaler.step(optimizer)
        scaler.update()
    print('amp cast to float:', time() - st)

import torch
import time

niter = 1000
# mode = 'trilinear'
mode = 'bilinear'

print(f'{mode=}')

# x = torch.randn((2, 512, 4, 4, 4), dtype=torch.float, device='cuda')
# upshape = (64, 64, 64)
x = torch.randn((2, 512, 4, 4), dtype=torch.float, device='cuda')
upshape = (64, 64)

# warmup
for _ in range(niter):
    y = torch.nn.functional.interpolate(x, upshape, None, mode)


# float
torch.cuda.synchronize()
start = time.time()

for _ in range(niter):
    y = torch.nn.functional.interpolate(x, upshape, None, mode)

torch.cuda.synchronize()
end = time.time()
print(f'float {(end - start) / niter : .4e}')


# half
xh = x.half()

torch.cuda.synchronize()
start = time.time()

for _ in range(niter):
    y = torch.nn.functional.interpolate(xh, upshape, None, mode)

torch.cuda.synchronize()
end = time.time()
print(f'half {(end - start) / niter : .4e}')

import torch

x = torch.randn(2, 512, 4, 4, 4, device='cuda')

y = torch.nn.functional.interpolate(x, (64, 64, 64), None, 'trilinear')

y.backward(torch.randn_like(y))

from time import time

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional
from torch.optim import SGD


class Network(nn.Module):
    def __init__(self, cast_for_upsample=False, conv_op=nn.Conv3d, stride=10, skip_upsample=False):
        """
        :param cast_for_upsample: if True, tensor will be cast to float before applying nn.functional.interpolate and
        then cast to half after that
        :param conv_op: nn.Conv3d or nn.Conv2d
        :param stride: stride of conv, just to make sure the output is smaller than the input
        :param skip_upsample: if True, the output will not be upsampled prior to pooling -> the problematic
        operation is skipped
        """
        super().__init__()
        self.skip_upsample = skip_upsample
        self.cast_for_upsample = cast_for_upsample

        self.layers = conv_op(1, 32, 3, stride, 1, 1, 1, False)

        if conv_op == nn.Conv2d:
            self.interp_mode = 'bilinear'
        else:
            self.interp_mode = 'trilinear'

    def forward(self, x):
        conv_output = self.layers(x)

        if not self.skip_upsample:
            if self.cast_for_upsample:
                conv_output = conv_output.float()

            # setting align_corners=False to suppress the warning. This has no effect
            conv_output = nn.functional.interpolate(conv_output, x.shape[2:], None, self.interp_mode, align_corners=False)

            if self.cast_for_upsample:
                conv_output = conv_output.half()

        pooled = conv_output.mean()
        return pooled


def run(inp, conv_op, skip_upsample=False):
    """
    runs 10 warmup iterations followed by 250 timed iterations
    uses float
    loss is just the mean of all outputs. It doesn't do anything useful. All we need is gradients
    """
    torch.cuda.empty_cache()

    net = Network(cast_for_upsample=False, conv_op=conv_op, skip_upsample=skip_upsample).cuda()
    optimizer = SGD(net.parameters(), 0.001)

    # warmup
    for _ in range(10):
        optimizer.zero_grad()
        out = net(inp)
        out.backward()
        optimizer.step()

    # fp32 measurement
    st = time()
    for _ in range(250):
        optimizer.zero_grad()
        out = net(inp)
        out.backward()
        optimizer.step()
    print('fp32:', time() - st)


def run_amp(inp, conv_op, cast_to_float_for_upsample, skip_upsample=False):
    """
    runs 10 warmup iterations followed by 250 timed iterations
    uses autocast
    loss is just the mean of all outputs. It doesn't do anything useful. All we need is gradients
    """
    torch.cuda.empty_cache()

    net = Network(cast_for_upsample=cast_to_float_for_upsample, conv_op=conv_op, skip_upsample=skip_upsample).cuda()
    optimizer = SGD(net.parameters(), 0.001)
    scaler = GradScaler()

    torch.cuda.empty_cache()

    # warmup
    for _ in range(10):
        optimizer.zero_grad()

        with autocast():
            out = net(inp)

        scaler.scale(out).backward()
        scaler.step(optimizer)
        scaler.update()

    # amp measurement
    st = time()
    for _ in range(250):
        optimizer.zero_grad()

        with autocast():
            out = net(inp)

        scaler.scale(out).backward()
        scaler.step(optimizer)
        scaler.update()
    print('amp, cast_to_float_for_upsample %s:' % cast_to_float_for_upsample, time() - st)


if __name__ == "__main__":
    ####### 2D ###########
    print('##### 2D #####')
    inp = torch.rand((1, 1, 512, 512)).cuda()
    conv_op = nn.Conv2d

    run(inp, conv_op)
    run_amp(inp, conv_op, False)
    run_amp(inp, conv_op, True)

    ####### 2D ###########
    print('##### 3D #####')
    inp = torch.rand((1, 1, 64, 64, 64)).cuda()
    conv_op = nn.Conv3d

    run(inp, conv_op)
    run_amp(inp, conv_op, False)
    run_amp(inp, conv_op, True)

    # now we skip the nn.functional.interpolate operation to demonstrate that this is not related to any other
    # component in the network
    ####### 2D ###########
    print('##### 2D, skipped nn.functional.interpolate #####')
    inp = torch.rand((1, 1, 512, 512)).cuda()
    conv_op = nn.Conv2d

    run(inp, conv_op, skip_upsample=True)
    run_amp(inp, conv_op, False, skip_upsample=True)

    ####### 2D ###########
    print('##### 3D, skipped nn.functional.interpolate no upsampling #####')
    inp = torch.rand((1, 1, 64, 64, 64)).cuda()
    conv_op = nn.Conv3d

    run(inp, conv_op, skip_upsample=True)
    run_amp(inp, conv_op, False, skip_upsample=True)