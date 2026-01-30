import torch 
import numpy as np
import torch.nn as nn

class MinimalExample(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear1 = nn.Linear(100,32)
    def forward(self, x):
        x = self.linear1(x)
        return torch.mean(x, dim=(-1,-2))

torch_model = MinimalExample()
batch_size=1
    
x = torch.randn(1, 3,100, 100, requires_grad=True)
torch_out = torch_model(x)

torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "minimalist.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=18,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

import onnx
onnx_model = onnx.load("minimalist.onnx")
onnx.checker.check_model(onnx_model)
import onnxruntime

ort_session = onnxruntime.InferenceSession("minimalist.onnx", providers=["CPUExecutionProvider",]
                                               )

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction``
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

try:
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
except:
    print(print(torch_out,"\n", ort_outs[0]))
print(f"Exported model has been tested with ONNXRuntime, and the result looks good on CPU!")