import torch.nn as nn

import torch
import time
import subprocess

def benchmark(device, dtype):
    # Create example inputs
    x = torch.testing.make_tensor(3, 5, 65536, device=device, dtype=dtype)
    sf = .5

    # Check output
    y = torch.nn.functional.interpolate(x, scale_factor=sf, mode="linear")
    z = torch.nn.functional.interpolate(x.cpu(), scale_factor=sf, mode="linear")
    outputs_match = torch.allclose(y.cpu(), z)
    if not outputs_match:
       atol = (y.cpu() - z).abs().max()
       rtol = ((y.cpu() - z)[z!=0]/z[z!=0]).abs().max()
       print(f"atol={atol} rtol={rtol}")

    # Measure time manually
    start_time = time.time() * 1000
    for _ in range(1000):
        y = torch.nn.functional.interpolate(x, scale_factor=sf, mode="linear")
    torch.mps.synchronize
    end_time = time.time() * 1000
    manual_delta = (end_time - start_time)
    average_time = f"{manual_delta:6.1f}"

    return "True " if outputs_match else "False", average_time

outputs_match_list = []
average_time_list = []
for device in ["mps", "cpu"]:
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        outputs_match, average_time = benchmark(device, dtype)
        outputs_match_list.append(str(outputs_match))
        average_time_list.append(average_time)

brand_string = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode("utf-8").strip()
print(f"\nBenchmarking Results (collected on {brand_string}):")
print("-"*40)
print("Device            :                MPS        |               CPU")
print("Dtype             :   FP32  |  FP16  |  BF16  |  FP32  |  FP16  |  BF16  ")
print(f"Outputs Match     :  ", " |  ".join(outputs_match_list))
print(f"Average Time (us) :", "  |".join(average_time_list))