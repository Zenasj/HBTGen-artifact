import torch.nn as nn

import torch

class PermuteModule(torch.nn.Module):
    def __init__(self, permutation):
        super(PermuteModule, self).__init__()
        self.permutation = permutation
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == len(self.permutation), f"Dimension mismatch! Unable to permute {len(x.shape)} dim input with a {len(self.permutation)} dim permutation!"
        return x.permute(*self.permutation)

def test(n_layers:int, conv_stride:int):
    _sequence = []
    for _ in range(n_layers):
        # Conv1d inputs are (N x C x L), LayerNorm expects (* x C). Dims must be permuted between modules.
        _sequence += [
            PermuteModule((0,2,1)),
            torch.nn.Conv1d(in_channels=512, out_channels=512, groups=1, kernel_size=9, dilation=1, stride=conv_stride, padding=0, bias=False),
            PermuteModule((0,2,1)),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU()
        ]
    model = torch.nn.Sequential(*_sequence).to(device="cuda")
    data = torch.randn((100,2048,512), device="cuda")
    out = model(data)
    loss = torch.nn.functional.mse_loss(out, torch.rand_like(out))
    loss.backward()

torch.autograd.set_detect_anomaly(True)
print(f"Torch version: {torch.__version__}")

print(f"layers=1, stride=1")
test(n_layers=1, conv_stride=1)
print(f"layers=2, stride=1")
test(n_layers=2, conv_stride=1)
print(f"layers=1, stride=2")
test(n_layers=1, conv_stride=2)
print(f"layers=2, stride=2")
test(n_layers=2, conv_stride=2)

# we will not reach this print statement.
print("DONE.")