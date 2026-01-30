import torch
import torch.nn as nn

class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    @torch.jit.script_if_tracing
    def onnx_helper(x_expanded: List[torch.Tensor]):
        temp_tensor = torch.stack(x_expanded)
        return temp_tensor

    def forward(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        dur[dur < 0] = 0.
        x_expanded = []
        for i in range(x.size(0)):
            x_exp = torch.repeat_interleave(x[i], (dur[i] + 0.5).long(), dim=0)
            x_expanded.append(x_exp)

        output = self.onnx_helper(x_expanded)
        return output