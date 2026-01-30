import torch
import fire
from torch._inductor.utils import do_bench_using_profiling

torch.manual_seed(0)

def get_max(x):
    x = torch.max(x)
    return x

def run(is_contiguous: bool = True):
    x = torch.randn(4096, 8192, dtype=torch.bfloat16, device="cuda")

    if not is_contiguous:
        x = x.t().contiguous().t()

    get_max_c = torch.compile(get_max)

    # warmup
    y = get_max_c(x)

    # perf
    duration_microseconds = do_bench_using_profiling(lambda: get_max_c(x))
    print('duration in microseconds', duration_microseconds)

if __name__ == '__main__':
    fire.Fire(run)