import torch.nn as nn

import torch
print(f"Torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
torch._dynamo.config.verbose=True
class MyMod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_stuff = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_stuff(x)
        return torch.nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
m = MyMod()
c = torch.compile(m, backend="aot_ts_nvfuser")
ex = torch.randn(1,32,16,16)
with torch.inference_mode():
    print(f"Base: {m(ex).shape}")
    print(f"Compiled: {c(ex).shape}")
print(f"Base: {m(ex).shape}")
print(f"Compiled: {c(ex).shape}")