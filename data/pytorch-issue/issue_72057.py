import torch
import torch.nn as nn
import numpy as np

norm = nn.InstanceNorm2d(64, affine=True, track_running_stats=True)
input = torch.randn(1, 64, 128, 128)
norm(input)

norm.eval()
torch.onnx._export(norm,             # model being run
                           (torch.rand(1,64, 128, 128)),                   
                           "./norm.onnx") ;
with torch.no_grad():
    torchout=norm(input)

import onnxruntime

ort_session = onnxruntime.InferenceSession("./norm.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(None,ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torchout), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")