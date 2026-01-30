import torch.nn.functional as F

import torch
from torch._inductor.runtime.benchmarking import benchmarker
from torch.nn import functional as F

def run(seqlen):
    with torch.device("cuda"):
        def f(q, k, v, mask):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0
            )
        f_compiled = torch.compile(f)

        # Create inputs
        bsz = 32
        q = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16)
        k = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16)
        v = torch.randn(bsz, 16, seqlen, 64, dtype=torch.bfloat16)
        mask = torch.ones([bsz, 1, seqlen, seqlen], dtype=torch.bool)
        inputs = [q, k, v, mask]

        # Benchmark
        time = benchmarker.benchmark_gpu(lambda: f_compiled(*inputs), warmup=5, rep=50)
        return time

for seqlen_start in [1008, 1024, 2048, 4096]:
    for offset in range(-1, 2):
        seqlen = seqlen_start + offset

        torch._dynamo.reset()
        time = run(seqlen)

        print(seqlen, time)
    print()