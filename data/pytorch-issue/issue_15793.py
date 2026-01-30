import torch.nn as nn

import torch
import time
import matplotlib.pyplot as plt


class RankSeparable(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            bias=True,
            stride=1)

        self.conv2 = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size - 1) // 2),
            bias=True,
            stride=1)

    def forward(self, in_tensor):
        output = self.conv1(in_tensor)
        output = self.conv2(output)
        return output


if __name__ == '__main__':

    torch.cuda.set_device(0)

    num_channels = 256
    batch = torch.rand(1, 256, 256, 512)
    batch = batch.cuda()

    separable = RankSeparable(num_channels, kernel_size=3)
    separable = separable.cuda()
    separable = separable.eval()

    times = [] 
    num_runs = 2000
    skip = 1  # We will just skip the first execution because it is usually slower
    with torch.no_grad():
        avg_time = 0
        for i in range(num_runs):
            start = time.time()
            separable.forward(batch)
            end = time.time()
            #torch.cuda.empty_cache() #this just increases the execution time
            if (i >= skip):
                elapsed = (end - start) * 1000
                avg_time += elapsed
                times.append(elapsed)
        avg_time /= (num_runs - skip)
        print("avg_time=={}ms, num_runs=={}".format(avg_time, num_runs))
    plt.plot(times)
    plt.show()