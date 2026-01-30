import torch.nn as nn

import timeit

import torch


def grid_sample_test(input, grid, backward):
    if backward and grid.grad is not None:
        grid.grad.zero_()
    samples = torch.nn.functional.grid_sample(
        input,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    m = samples.mean()
    if backward:
        m.backward()

    return samples


_input = None
_grid = None
_backward = None

if __name__ == "__main__":
    torch.manual_seed(15)
    torch.set_num_threads(1)

    N = 100
    C = 2
    repeats = 100
    H_out = 13
    W_out = 13
    dtype = torch.double
    devices = ["cpu"]
    backwards = [False, True]

    input_sizes = [(30, 40), (300, 400), (1000, 1200)]

    grid_cpu = 2.0 * torch.rand((N, H_out, W_out, 2), dtype=dtype) - 1.0

    for input_size in input_sizes:
        H_in, W_in = input_size
        input_cpu = torch.rand(
            (1, C, H_in, W_in),
            requires_grad=False,
            dtype=dtype,
        ).expand((N, -1, -1, -1))

        for _backward in backwards:
            for device in devices:
                _grid = grid_cpu.clone().detach().to(device).requires_grad_(True)
                _input = input_cpu.to(device)

                t = timeit.timeit(
                    "grid_sample_test(_input, _grid, _backward)",
                    globals=globals(),
                    number=repeats,
                )
                print(
                    f"device={device:>4} backward={str(_backward):>5} input size={H_in:>4}x{W_in:<4}: {t:5.2f}"
                )