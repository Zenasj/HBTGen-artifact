import torch.nn as nn

import torch
import time
import subprocess
import itertools

def benchmark(device, dtype, mode="bilinear", antialias=False, sf=.5):
    # Create example inputs
    x = torch.testing.make_tensor(1, 1, 2048, 2048, device=device, dtype=dtype)

    # define kwargs
    kwargs = {"antialias": antialias, "mode": mode, "scale_factor": sf}

    # Skip for unimplemented flavors
    if antialias and mode == "bicubic" and device == "mps":
       return None, "Skip"
    elif antialias and dtype != torch.float32:
       if device == "cpu":
           return None, "Skip"
       outputs_match = None
    else:
        # Check output
        y = torch.nn.functional.interpolate(x, **kwargs)
        z = torch.nn.functional.interpolate(x.cpu(), **kwargs)
        outputs_match = torch.allclose(y.cpu(), z)
        if not outputs_match:
           atol = (y.cpu() - z).abs().max()
           rtol = ((y.cpu() - z)[z!=0]/z[z!=0]).abs().max()
           print(f"atol={atol} rtol={rtol}")

    # Measure time manually
    start_time = time.time() * 1000
    for _ in range(1000):
        y = torch.nn.functional.interpolate(x, **kwargs)
    torch.mps.synchronize()
    end_time = time.time() * 1000
    manual_delta = (end_time - start_time)
    average_time = f"{manual_delta:6.1f}"

    return "True " if outputs_match else "False", average_time

brand_string = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode("utf-8").strip()
for mode,antialias in itertools.product(["bilinear", "bicubic"], [False, True]):
    outputs_match_list = []
    average_time_list = []
    for device in ["mps", "cpu"]:
      for dtype in [torch.float32, torch.float16, torch.bfloat16]:
          outputs_match, average_time = benchmark(device, dtype, mode=mode, antialias=antialias)
          outputs_match_list.append(str(outputs_match))
          average_time_list.append(average_time)

    print(f"\nBenchmarking Results (collected on {brand_string}) for {mode} interpolation {'with antialias' if antialias else ''}:")
    print("-"*40)
    print("Device            :                MPS        |               CPU")
    print("Dtype             :   FP32  |  FP16  |  BF16  |  FP32  |  FP16  |  BF16")
    print(f"Outputs Match     :  ", " |  ".join(outputs_match_list))
    print(f"Average Time (us) :", "  |".join(average_time_list))