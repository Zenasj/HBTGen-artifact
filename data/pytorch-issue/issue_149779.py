import torch.nn.functional as F

import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

timing = []
batches=  list(range(32, 4096, 32))

for batch in [32] + batches:
    samples = []
    for _ in range(100):
        probs = torch.rand(batch, 10).cuda()
        labels = torch.randint(0, 10, (batch,)).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        F.nll_loss(probs, labels)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        samples.append(elapsed)
    timing.append(sum(samples) / len(samples))
timing = timing[1:]

plt.plot(batches, timing)
plt.show()