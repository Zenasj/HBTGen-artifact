import torch

with self.assertRaises(AttributeError):
    torch.xpu.is_available()  # type: ignore[attr-defined]