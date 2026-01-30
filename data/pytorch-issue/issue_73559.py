import torch
import torch.nn as nn
import onnxruntime as ort

class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
    def forward(self, x):
        return torch.linspace(-1.0, 1.0, x[0,0])

model = SimpleModule()
model.eval()
x = torch.IntTensor([[6]])
torch_out = model(x)
torch.onnx.export(model, x, "test.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})
s = ort.InferenceSession("test.onnx")
output = s.run(None, input_feed = {"input" : [[5]]})
print(len(output[0]), output[0])
output = s.run(None, input_feed = {"input" : [[6]]})
print(len(output[0]), output[0])
output = s.run(None, input_feed = {"input" : [[7]]})
print(len(output[0]), output[0])

torch_output = model(torch.IntTensor([[5]]))
print(len(torch_output), torch_output)
torch_output = model(torch.IntTensor([[6]]))
print(len(torch_output), torch_output)
torch_output = model(torch.IntTensor([[7]]))
print(len(torch_output), torch_output)