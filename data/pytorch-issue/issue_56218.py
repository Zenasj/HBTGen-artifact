import torch
import torch.nn as nn

input_tensor = torch.rand(1, 3, 224, 224)
opset_version = 10

# Dummy model

class DummyModel(nn.Module):

    def __init__(self):

        super(DummyModel, self).__init__()

        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):

        x = self.avg_pool2d(x)

        return x

dummy_model = DummyModel()
dummy_model.eval()
onnx_file_path = "dummy_model.onnx"
torch.onnx.export(model=dummy_model,
                    args=input_tensor,
                    f=onnx_file_path,
                    do_constant_folding=True,
                    opset_version=opset_version,
                    export_params=True,
                    )