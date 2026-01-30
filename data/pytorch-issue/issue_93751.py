import contextlib
import traceback
import time

import torch
import torchvision
import torchdynamo
from torchdynamo.optimizations.training import aot_autograd_speedup_strategy

N_WARMUP = 100
N_BENCH = 100

def bench(batch_size, use_dynamo):
    model = torchvision.models.resnet50().cuda()
    x = torch.randn(batch_size, 3, 224, 224, dtype=torch.float, device='cuda')

    train_context = torchdynamo.optimize(aot_autograd_speedup_strategy) if use_dynamo is True else contextlib.nullcontext()

    torch.cuda.synchronize()
    t0  = time.time()

    with train_context:
        for _ in range(N_WARMUP):
            out = model(x)
            out.sum().backward()

        torch.cuda.synchronize()
        t1 = time.time()

        for _ in range(N_BENCH):
            out = model(x)
            out.sum().backward()

        torch.cuda.synchronize()
        t2 = time.time()

    print('Training img/s (larger better):', batch_size / ((t2 - t1) / N_BENCH))
    print('Total time incl. overhead (smaller better):', t2 - t0)
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--use_dynamo', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    bench(args.batch_size, args.use_dynamo)