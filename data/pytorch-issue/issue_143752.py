import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
from torch._inductor import config

config.fallback_random = True


class Model(torch.nn.Module):

    def __init__(self, pool_operator):
        super(Model, self).__init__()
        self.pool = pool_operator

    def forward(self, x):
        x = torch.argmax(x, dim=1)
        # when touching here, x.dtype=torch.int64
        x = self.pool(x)
        return x


def run_test(dim, device, backend):
    op_inst = eval(f"nn.AdaptiveMaxPool{dim}d(5)")
    model = Model(op_inst).to(device)
    x = torch.randn([1] * (dim + 2)).to(device)

    if backend == "inductor":
        model = torch.compile(model)

    try:
        y = model(x)
        print(f"succeed on {device} with {backend}: {y.dtype}")
    except Exception as e:
        print(f"fail on {device} with {backend}: {e}")


run_test(1, "cpu", "eager")  # fail on cpu with eager: "adaptive_max_pool2d" not implemented for 'Long'
run_test(1, "cpu", "inductor")  # succeed on cpu with inductor: torch.int64
run_test(1, "cuda", "eager")  # fail on cuda with eager: "adaptive_max_pool2d_cuda" not implemented for 'Long'
run_test(1, "cuda", "inductor")  # fail on cuda with inductor: backend='inductor' raised: SubprocException: An exception occurred in a subprocess:


run_test(2, "cpu", "eager")  # fail on cpu with eager: "adaptive_max_pool2d" not implemented for 'Long'
run_test(2, "cpu", "inductor")  # succeed on cpu with inductor: torch.int64
run_test(2, "cuda", "eager")  # fail on cuda with eager: "adaptive_max_pool2d_cuda" not implemented for 'Long'
run_test(2, "cuda", "inductor")  # # fail on cuda with inductor: backend='inductor' raised: SubprocException: An exception occurred in a subprocess:


run_test(3, "cpu", "eager")  # fail on cpu with eager: "adaptive_max_pool3d_cpu" not implemented for 'Long'
run_test(3, "cpu", "inductor")  # fail on cpu with inductor: "adaptive_max_pool3d_cpu" not implemented for 'Long'
run_test(3, "cuda", "eager")  # fail on cuda with eager: "adaptive_max_pool3d_cuda" not implemented for 'Long'
run_test(3, "cuda", "inductor")  # fail on cuda with inductor: "adaptive_max_pool3d_cuda" not implemented for 'Long'