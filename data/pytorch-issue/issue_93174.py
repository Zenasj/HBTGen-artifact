torch.onnx.export(module, fake_input_tensors, "/home/yyy/out.onnx")

from torch import jit
model = jit.load('/tmp/varad_409140779.pt')
import torch

device='cpu'

# following fake_input copied from dec_model_v2_test.py
fake_input_tensors=(
            torch.tensor([1000, 1001], dtype=torch.int64, device=device),
            torch.tensor(
                    [101, 101], dtype=torch.int64, device=device
                    ),
            torch.tensor([2, 2], dtype=torch.int32, device=device),
            torch.tensor(
                    [1, 2, 3, 4], dtype=torch.int32, device=device
                    ),
            torch.tensor(
                    [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device=device
                    ),
            torch.tensor(
                    [0], dtype=torch.int32, device=device
                    ),
            torch.tensor(
                    [], dtype=torch.int32, device=device
                    ),
            torch.tensor(
                    [], dtype=torch.int64, device=device
                    ),
            torch.tensor([1, 1], dtype=torch.int32, device=device),
            torch.tensor([1, 0], dtype=torch.int32, device=device),
            torch.tensor([1, 1], dtype=torch.int32, device=device),
            torch.tensor([1, 2], dtype=torch.int32, device=device),
            torch.tensor(
                    [0.9, 0.8], dtype=torch.float32, device=device
                    ),

 torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.randn(
                    (2, 128), dtype=torch.float32, device=device
                    ),
            torch.tensor(
                    [100, 300, 200], dtype=torch.int32, device=device
                    ),
            torch.tensor(
                    [16, 64, 32], dtype=torch.int32, device=device
                    ),
            torch.tensor(
                    [2, 1, 2], dtype=torch.int32, device=device
                    ),
            torch.tensor(
                    [0, 1, 0, 0, 1], dtype=torch.int32, device=device
                    ),
            torch.arange(
                    2 * 16 + 1 * 64 + 2 * 32, dtype=torch.float32, device=device
                    ),
            )

output = model(*fake_input_tensors)

torch.onnx.export(model, fake_input_tensors, "/tmp/out.onnx")

@torch.jit.script
def length_to_indices(lengths: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(
            start=0, end=lengths.size()[0], device=lengths.device, dtype=torch.int64
        ),
        lengths.to(dtype=torch.int64),
    )