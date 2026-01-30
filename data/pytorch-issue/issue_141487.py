python
import torch

test_inputs = [
    torch.tensor([complex(torch.inf, 0), complex(0, torch.inf), complex(torch.inf, torch.inf)], dtype=torch.complex128),
]
test_apis = [
    torch.sin, torch.cos, torch.tan, torch.acos, torch.asin, torch.atan,
    torch.sinh, torch.cosh, torch.tanh, torch.exp, torch.rsqrt, torch.mean
]

for api in test_apis:
    print(f"Testing {api.__name__}")
    for x in test_inputs:
        try:
            cpu_out = api(x)
            gpu_out = api(x.cuda())
            print(f"CPU Output: {cpu_out}")
            print(f"GPU Output: {gpu_out}")
        except Exception as e:
            print(f"Error in {api.__name__}: {e}")