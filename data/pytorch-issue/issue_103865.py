import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        return out

with torch.onnx.enable_fake_mode() as fake_context:
    x = torch.rand(5, 2, 2)
    model = Model()

# Export the model with fake inputs and parameters
export_options = ExportOptions(fake_context=fake_context)
export_output = torch.onnx.dynamo_export(
    model, x, export_options=export_options
)

model_state_dict = Model().state_dict()  # optional
export_output.save("/path/to/model.onnx", model_state_dict=model_state_dict)