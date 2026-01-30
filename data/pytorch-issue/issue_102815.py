import sys

import torch

l_inputs = [
    ((1024,), 0, 2, 100),
    ((4096,), 0, 2, 100),
    ((16384,), 0, 4, 100),
    ((32000,), 0, 8, 100),
    ((128 * 1024,), 0, 2, 100),
    ((256 * 1024,), 0, 3, 100),
    ((1 * 1024 * 1024,), 0, 2, 100),
    ((4 * 1024 * 1024,), 0, 2, 100),
    ((16 * 1024 * 1024,), 0, 2, 100),
    ((32 * 1024 * 1024,), 0, 2, 100),
    ((128 * 1024 * 1024,), 0, 2, 50),
    ((64, 256), 0, 4, 100),
    ((400, 400), 0, 2, 100),
    ((640, 1080), 0, 2, 100),
    ((128, 4096), 1, 2, 100),
    ((512, 512), 1, 2, 100),
    ((699, 713), 1, 2, 100),
    ((1024, 1024), 1, 2, 100),
    ((2000, 1000), 1, 2, 100),
    ((4096, 4096), 1, 2, 100),
    ((16384, 16384), 1, 2, 50),
    ((384, 256, 16), 1, 2, 100),
    ((400, 200, 13), 1, 2, 100),
    ((128, 64, 256), 0, 2, 100),
    ((512, 256, 256), 1, 2, 100),
    ((512, 1024, 1024), 2, 2, 10),
    ((1024, 512, 1024), 2, 2, 10),
    ((1024, 1024, 512), 2, 2, 10),
    ((128, 64, 64, 32), 0, 2, 50),
    ((128, 64, 128, 16), 1, 2, 50),
    ((100, 45, 45, 32), 3, 2, 50),
    ((128, 32, 256, 32), 3, 2, 50),
]

prof_inputs = [
    ((1234567,), 0, 2, 5),
    ((16 * 1024 * 1024,), 0, 3, 5),
    ((1013, 1013), 0, 2, 5),
    ((1024, 1024), 1, 2, 5),
    ((69, 74, 128), 0, 2, 5),
    ((128, 128, 128), 2, 2, 5),
]


def generate_tensors(dim_tuple, cat_type, num_tensors):
    if cat_type in [torch.int8, torch.int32, torch.int64]:
        l_tensors = [
            torch.randint(
                high=torch.iinfo(cat_type).max,
                size=dim_tuple,
                dtype=cat_type,
                device="cuda",
            )
        ] * num_tensors
        return l_tensors
    else:
        l_tensors = [
            torch.randn(dim_tuple, dtype=cat_type, device="cuda")
        ] * num_tensors
        return l_tensors


def test_simple_cat(
    dim_tuple, cat_dim: int, num_tensors: int, iterations: int, cat_type
):
    torch.cuda.synchronize()

    # Allocate a tensor equal to L2 cache size on A100 GPUs
    l2_cache_flusher = torch.empty(
        int(80 * (1024**2)), dtype=torch.float, device="cuda"
    )

    # All the tensors in the list get read and written once
    total_MB = 2 * num_tensors
    for dim in dim_tuple:
        total_MB *= dim
    total_MB /= 1024 * 1024

    # Get the number of bits per element
    if cat_type in [torch.int8, torch.int32, torch.int64]:
        total_MB *= torch.iinfo(cat_type).bits / 8
    else:
        total_MB *= torch.finfo(cat_type).bits / 8

    l_tensors = generate_tensors(dim_tuple, cat_type, num_tensors)
    c = torch.cat(l_tensors, dim=cat_dim)
    torch.cuda.synchronize()

    # Measure correctness
    l_tensors_cpu = []
    for t in l_tensors:
        l_tensors_cpu.append(t.detach().to("cpu"))
    c_cpu = torch.cat(l_tensors_cpu, dim=cat_dim)
    c_cpu_dev = c.detach().to("cpu")

    if not torch.equal(c_cpu, c_cpu_dev):
        missmatches = torch.count_nonzero(torch.abs(c_cpu - c_cpu_dev))
        print("Error; num missmatches for {0} = {1}".format(dim_tuple, missmatches))
        return

    # Measure a few iterations
    l_ev_start = [torch.cuda.Event(enable_timing=True)] * iterations
    l_ev_stop = [torch.cuda.Event(enable_timing=True)] * iterations

    l_cat_times = []
    torch.cuda.synchronize()
    for i in range(iterations):
        l2_cache_flusher.zero_()
        torch.cuda._sleep(1_000_000)

        l_ev_start[i].record()
        c = torch.cat(l_tensors, dim=cat_dim)
        l_ev_stop[i].record()
    torch.cuda.synchronize()

    for i in range(iterations):
        t_cat = l_ev_start[i].elapsed_time(l_ev_stop[i]) / 1000
        l_cat_times.append(t_cat)

    min_cat_time = min(l_cat_times)

    # return bandwidth in GB/s
    estimated_bw_GBps = total_MB / min_cat_time / 1024
    return estimated_bw_GBps


def main(argv):
    if len(argv) > 0:
        if "profile" in str(argv[0]):
            for l_input in prof_inputs:
                gbps = test_simple_cat(
                    l_input[0], l_input[1], l_input[2], l_input[3], torch.float
                )
                print(
                    "Bandwidth (GB/s) for {0} fp32 | {1:.2f}".format(
                        (l_input[0], l_input[1]), gbps
                    )
                )
            return

    for l_input in l_inputs:
        gbps_int8 = test_simple_cat(
            l_input[0], l_input[1], l_input[2], l_input[3], torch.int8
        )
        gbps_fp16 = test_simple_cat(
            l_input[0], l_input[1], l_input[2], l_input[3], torch.float16
        )
        gbps_fp32 = test_simple_cat(
            l_input[0], l_input[1], l_input[2], l_input[3], torch.float32
        )
        gbps_int32 = test_simple_cat(
            l_input[0], l_input[1], l_input[2], l_input[3], torch.int32
        )
        gbps_fp64 = test_simple_cat(
            l_input[0], l_input[1], l_input[2], l_input[3], torch.float64
        )
        gbps_long = test_simple_cat(
            l_input[0], l_input[1], l_input[2], l_input[3], torch.long
        )

        print(
            "Bandwidth (GB/s) for {0} int8;fp16;fp32;int32;fp64;long|{1:.2f}|{2:.2f}|{3:.2f}|{4:.2f}|{5:.2f}|{6:.2f}".format(
                (l_input[0], l_input[1]),
                gbps_int8,
                gbps_fp16,
                gbps_fp32,
                gbps_int32,
                gbps_fp64,
                gbps_long,
            )
        )


if __name__ == "__main__":
    main(sys.argv[1:])