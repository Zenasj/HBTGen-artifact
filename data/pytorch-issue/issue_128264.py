import torch
with torch.inference_mode():
    print(torch.autograd.functional.jacobian(lambda x: x**2, torch.tensor(1.0)))