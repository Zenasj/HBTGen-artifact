import torch.nn as nn

import torch
import torch.utils.benchmark as benchmark

# exit cleanly if we are on a device that doesn't support torch.compile
if torch.cuda.get_device_capability() < (7, 0):
    print("Exiting because torch.compile is not supported on this device.")
    import sys
    sys.exit(0)


# Let's define a helpful benchmarking function:
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

    
def main():
    model = torch.nn.Sequential(
        *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
    )
    input = torch.rand(1024, device="cuda")
    output = model(input)
    output.sum().backward()

    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    @torch.compile(fullgraph=False)
    def fn():
        opt.step()

    # Warmup runs to compile the function
    for _ in range(5):
        fn()

    eager_runtime = benchmark_torch_function_in_microseconds(opt.step)
    compiled_runtime = benchmark_torch_function_in_microseconds(fn)

    assert eager_runtime > compiled_runtime

    print(f"eager runtime: {eager_runtime}us")
    print(f"compiled runtime: {compiled_runtime}us")

if __name__ == '__main__':
    main()