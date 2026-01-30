import torchvision

python
# - With torch 1.5.1, torchvision 0.6.1 and onnx 1.7.0, this script raises the runtime error shown below at rep.run()
# - With torch 1.4, torchvision 0.5 and onnx 1.7.0 it runs without issues.

import onnx
import torch
from caffe2.python.onnx import backend

from torchvision.models import resnet50

rn50 = resnet50(pretrained=True)

# Export the model
dummy_input = torch.randn(1, 3, 224, 224).cpu()
rn50.cpu()
torch.onnx.export(rn50, dummy_input, "/tmp/rn50_test.onnx", keep_initializers_as_inputs=True, verbose=False)

# Load and test that the export went correctly
onnx_model = onnx.load("/tmp/rn50_test.onnx")
onnx.checker.check_model(onnx_model)
rep = backend.prepare(onnx_model, device="CPU")

# the following rep.run() line raises:
# RuntimeError: [enforce fail at conv_pool_op_base.h:160] kernel_[dim]. If you are doing convolution or pooling, you will need to set explicitly the kernel size.
onnx_out = rep.run(dummy_input.numpy())