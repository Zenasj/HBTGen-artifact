import torch
import torch.nn as nn

class SingleLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(5, 5, 1).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv1d(x)
        return x

    def get_example_inputs(self):
        return (torch.rand(5, 5, 1),)

torch.backends.quantized.engine = "fbgemm"
model = SingleLayerModel()
example_input = model.get_example_inputs()[0]
q_config = q_config_mapping = QConfigMapping()
q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))

prepared = quantize_fx.prepare_fx(model, q_config_mapping, example_input)
print(prepared.conv1d.qconfig.weight.p.func)