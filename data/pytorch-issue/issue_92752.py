import platform
import torch

device = "mps"


def main() -> None:
    """Find bug."""
    probs = torch.tensor([[2.3634e-04, 1.2352e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                           0.0000e+00]], device="cpu")
    idx_next = torch.multinomial(probs, num_samples=1)
    print(idx_next)

    mps_probs = torch.tensor([[2.3634e-04, 1.2352e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                               0.0000e+00]], device="mps")
    mps_idx_next = torch.multinomial(mps_probs, num_samples=1)
    print(mps_idx_next)

    print(f"python_version: {platform.python_version()}")
    print(f"torch.__version__: {torch.__version__}")

    # tensor([[0]])
    # tensor([[-9223372036854775808]], device='mps:0')


if __name__ == "__main__":
    main()

import platform
print(f'{platform.python_version()=}')

import torch
print(f'{torch.__version__=}')

tensor = torch.Tensor([0, 100, 0]).float()

mps_tensor = tensor.to('mps')
mps_sample = torch.multinomial(mps_tensor, 1)

cpu_tensor = tensor.to('cpu')
cpu_sample = torch.multinomial(cpu_tensor, 1)

print(f'{mps_tensor=}')
print(f'{mps_sample=}')
print(f'{cpu_tensor=}')
print(f'{cpu_sample=}')