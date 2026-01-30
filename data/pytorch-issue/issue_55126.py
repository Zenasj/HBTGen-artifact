import timeit

import torch
from torch.utils.benchmark import Measurement, Timer

def make_timer_view():
    return Timer("x.view(-1);", setup="at::Tensor x=torch::empty({2,2} ); ", language="c++", timer=timeit.default_timer)

def make_timer_reshape():
    return Timer("x.reshape(-1);", setup="at::Tensor x=torch::empty({2,2} ); ", language="c++", timer=timeit.default_timer)


def measure_main(fn_make_timer):
    timer = fn_make_timer()
    counts = timer.collect_callgrind(number=20, repeats=5, collect_baseline=False)
    times = timer.blocked_autorange(min_run_time=5)
    return counts, times



def main(**kwargs):

    times = [[], []]
    counts = [[], []]
    c, t = measure_main(make_timer_view)
    times[0].append(t)
    counts[0].append(c)
    c, t = measure_main(make_timer_reshape)
    times[1].append(t)
    counts[1].append(c)

    print()

    t0 = times[0]
    t1 = times[1]
    
    # Take the min as any interpreter jitter will increase from the baseline.
    c0 = min(counts[0][0], key=lambda x: x.counts(denoise=True))
    c1 = min(counts[1][0], key=lambda x: x.counts(denoise=True))

    torch.set_printoptions(linewidth=200)
    delta = (
        c1.as_standardized().stats() -
        c0.as_standardized().stats()
    ).denoise()

    print(t0, "\n")
    print(t1, "\n")
    print(c0.counts(denoise=True))
    print(c1.counts(denoise=True), "\n")
    print(delta)

    # Uncomment to debug:
    # import pdb
    # pdb.set_trace()




if __name__ == "__main__":

    main()