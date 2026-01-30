import torch
import torch.nn as nn

input_tensor = torch.rand(1, 3, 2, 2)
opset_version = 10

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.layer = nn.AvgPool2d(
            kernel_size=(3, 3),
            stride=(3, 3),
            padding=(1,1),
            ceil_mode=True)
    def forward(self, x):
        x = self.layer(x)
        return x

dummy_model = DummyModel()
output_torch = dummy_model(input_tensor)
print('PyTorch input', input_tensor.shape)
print('PyTorch output', output_torch.shape)

onnx_file_path = "dummy_model.onnx"
torch.onnx.export(model=dummy_model,
                  args=input_tensor,
                  f=onnx_file_path,
                  do_constant_folding=True,
                  opset_version=opset_version,
                  input_names = ['input'],
                  output_names = ['output'],
                  export_params=True
                  )
import onnxruntime
onnxruntime_net = onnxruntime.InferenceSession(onnx_file_path)
print('onnxruntime input', onnxruntime_net.get_inputs()[0].shape)
print('onnxruntime output', onnxruntime_net.get_outputs()[0].shape)