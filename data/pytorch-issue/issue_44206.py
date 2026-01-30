import argparse
from time import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=8, out_channels=1024, kernel_size=(3, 3, 3), bias=False, padding=(1, 1, 1))
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


def main():
    flags = parse_arguments()
    torch.backends.cudnn.benchmark = True
    nvtx_enabled = False
    if flags.profile:
        import torch.cuda.profiler as profiler
        import pyprof
        pyprof.init(enable_function_stack=True)
        nvtx_enabled = True
        print("Profiling is enabled")

    model = Model()
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-8)
    scaler = GradScaler()

    data = torch.rand(size=(flags.batch_size, *flags.input_shape)).to(device)
    with torch.autograd.profiler.emit_nvtx(enabled=nvtx_enabled):
        model.train()
        iteration = 0
        start_marker = time()
        for i in range(flags.iterations):
            iteration += 1
            if flags.profile and iteration == 10:
                profiler.start()
            optimizer.zero_grad()

            with autocast(enabled=flags.amp):
                outputs = model(data)
                loss = torch.sum(outputs)

            if flags.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        if flags.profile:
            profiler.stop()
        elapsed = time() - start_marker
        print("Finished in", round(elapsed, 2),
              "s. One iteration took on average", round(elapsed/flags.iterations), "s.")


def parse_arguments():
    PARSER = argparse.ArgumentParser(description="UNet-3D")
    PARSER.add_argument('--amp', dest='amp', action='store_true', default=False)
    PARSER.add_argument('--batch_size', dest='batch_size', type=int, default=1)
    PARSER.add_argument('--input_shape', nargs='+', type=int, default=[32, 64, 64, 64])
    PARSER.add_argument('--profile', dest='profile', action='store_true', default=False)
    PARSER.add_argument('--iterations', dest='iterations', type=int, default=100)
    return PARSER.parse_args()


if __name__ == '__main__':
    main()