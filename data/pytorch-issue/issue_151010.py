import torch.nn as nn

import torch
from torch.export import Dim, export_for_training


class CustomLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=9, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)


batch = Dim("batch", min=2, max=None)
dynamic_shapes = ({0: batch},)

exported = export_for_training(CustomLSTM(), args=(torch.randn((128, 1, 9)),), strict=True, dynamic_shapes=dynamic_shapes)

a = exported.module()
b = a.to(torch.device("cuda:0"))
b.forward(torch.randn((128, 1, 9), device=torch.device("cuda:0")))

def _to(self, device: torch.device):
    raise NotImplementedError("Calling to() is not supported to move exported graph, use move_to_device_pass .")

module.to = types.MethodType(_to, module)  # type: ignore[method-assign]

def module(self) -> torch.nn.Module:
    """
    Returns a self-contained GraphModule with all the parameters/buffers inlined.
    """