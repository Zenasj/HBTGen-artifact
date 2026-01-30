import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import platform

if not torch.cuda.is_available():
    raise RuntimeError("This issue only happens on gpu")


def print_system_info():
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Platform: {platform.platform()}")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(x).squeeze(dim=-1)


def main(device: str, dtype: torch.dtype, dimension_size: int = 517):
    print(f"TESTING: {device} {dtype}")
    device = torch.device(device)
    model = TinyModel().to(device).to(dtype)
    optimizer = optim.Adam(model.parameters())

    x = (torch.randn((5, dimension_size, 5), dtype=dtype, device=device) - 0.5) * 2

    optimizer.zero_grad()
    logits = model(x)

    print("\nLogits info:")
    print(f"Shape: {logits.shape}")
    print(f"Device: {logits.device}")
    print(f"Dtype: {logits.dtype}")
    print(f"Requires grad: {logits.requires_grad}")

    # Test different softmax approaches
    print("\nSoftmax sums:")
    print("Method 1:", F.log_softmax(logits, dim=1).exp().sum(dim=1))

    # Try CPU version
    cpu_logits = logits.cpu()
    print("CPU version:", F.log_softmax(cpu_logits, dim=1).exp().sum(dim=1))

    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    print("Method 2:", log_probs.exp().sum(dim=1))


if __name__ == "__main__":
    print_system_info()
    # On my setup 517 was the minimum that caused the issue might be different on different machines
    dim_size = 517
    main("cuda", torch.float64, dim_size)   # Doesnt work
    # main("cuda", torch.float32, dim_size) # Works
    # main("cpu", torch.float64, dim_size)  # Works
    # main("cpu", torch.float32, dim_size)  # Works