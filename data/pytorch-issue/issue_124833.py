import random

import torch

torch.randn(1, device="mps")  # sample

with torch.random.fork_rng(device_type="mps"):  # fork rng
    ...

before_state = torch.mps.get_rng_state()
torch.randn(1, device="mps")  # sample again
after_state = torch.mps.get_rng_state()

print("RNG state progressed:", not torch.allclose(before_state, after_state))  # expected: True, actual: False