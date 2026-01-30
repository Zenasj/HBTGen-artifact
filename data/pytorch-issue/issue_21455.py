import math

if ceil_mode == 1:
    out_h = math.ceil(((in_h + pad_hl + pad_hr - kernel_size) / stride) + 1)
    out_w = math.ceil(((in_w + pad_wl + pad_wr - kernel_size) / stride) + 1)
else:
    out_h = int((in_h + pad_hl + pad_hr - kernel_size) / stride + 1)
    out_w = int((in_w + pad_wl + pad_wr - kernel_size) / stride + 1)

import torch

from models.faceboxes import FaceBoxes

img_dim = (720, 1280)
rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
model_path = "weights/Final_FaceBoxes.pth"
onnx_path = "weights/Final_FaceBoxes.onnx"

# Create the model and load the weights
model = FaceBoxes('test', img_dim, num_classes)

state_dict = torch.load(model_path)
model.load_state_dict(state_dict, strict=False)

# Create dummy input
dummy_input = torch.rand(1, 3, img_dim[0], img_dim[1])

# Define input / output names
input_names = ["actual_input_1"]
output_names = ["output1", "353"]

# Convert the PyTorch model to ONNX
torch.onnx.export(model,
                  dummy_input,
                  onnx_path,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

import torch

from models.faceboxes import FaceBoxes

img_dim = (720, 1280)
rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
model_path = "weights/Final_FaceBoxes.pth"
onnx_path = "weights/Final_FaceBoxes.onnx"

# Create the model and load the weights
model = FaceBoxes('test', img_dim, num_classes)

state_dict = torch.load(model_path)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

# Create dummy input
dummy_input = torch.rand(1, 3, img_dim[0], img_dim[1])

# Define input / output names
input_names = ["actual_input_1"]
output_names = ["output1", "353"]

# Convert the PyTorch model to ONNX
torch.onnx.export(model,
                  dummy_input,
                  onnx_path,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)