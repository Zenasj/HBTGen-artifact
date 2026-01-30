import torch.nn as nn
import numpy as np

import torch as ch
import torch.utils.benchmark as benchmark
import torch.autograd.profiler as profiler

def measure(tpe, cudnn):
    ch.backends.cudnn.enabled = cudnn
    model = ch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).cuda()
    inp = ch.zeros(512, 3, 32, 32).cuda()

    if tpe=='fp16':
        inp = inp.half()
        model = model.half()
    elif tpe=='fp64':
        inp = inp.double()
        model = model.double()

    def test():
        output = model(inp)
        loss = output.sum()
        loss.backward()

    timer = benchmark.Timer("""test()""", globals={'test': test})

    result = timer.timeit(1000) # warmup
    result = timer.timeit(1000) # real

    print(tpe, cudnn, result.median * 1000, "ms")

if __name__ == '__main__':
    measure('fp16', True)
    measure('fp32', True)
    measure('fp64', True)
    measure('fp16', False)
    measure('fp32', False)
    measure('fp64', False)