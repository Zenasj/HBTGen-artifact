import torch
import torch.nn as nn

class DictModule(torch.nn.Module):
    def forward(self, x_in: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        x_out = {}
        x_out["test_key_out"] = x_in
        return x_out
    
x_in = torch.tensor(1)
dms = torch.jit.script(DictModule())
torch.onnx.export(dms, (x_in,), "/dev/null", example_outputs=(dms(x_in),))