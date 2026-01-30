import torch.nn as nn

import torch

# ~/Documents/pytorch/data/dog.jpg
model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
model.eval()

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import pathlib
import tempfile
import torch.utils.mobile_optimizer

input_image = Image.open('~/Documents/pytorch/data/dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output[0], dim=0))

traced = torch.jit.trace(model, input_batch)
sum(p.numel() * p.element_size() for p in traced.parameters())
tf = pathlib.Path('~/Documents/pytorch/data/data/example_debug_map_with_tensorkey.ptl')

torch.jit.save(traced, tf.name)
print(pathlib.Path(tf.name).stat().st_size)
traced._save_for_lite_interpreter(tf.name)
print(pathlib.Path(tf.name).stat().st_size)
print(tf.name)

import torch
from torch.jit.mobile import _load_for_lite_interpreter
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

input_image = Image.open('~/Documents/pytorch/data/dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
reload_lite_model = _load_for_lite_interpreter('~/Documents/pytorch/experiment/example_debug_map_with_tensorkey.ptl')

with torch.no_grad():
    output_lite = reload_lite_model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output_lite[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(torch.nn.functional.softmax(output_lite[0], dim=0))

('constants',
    (0.1,
     1e-05,
     None,
     2,
     1,
     False,
     0,
     True,
     False,
     1,
     2,
     3,
     ('tensor_jit_index', 4),
     ('tensor_jit_index', 3),
     ('tensor_jit_index', 2),
     0.1,
     1e-05,
     None,
     2,
     1,
     False,
     0,
     24,
     True,
     58,
     ('tensor_jit_index', 0),
     3,
     ('tensor_jit_index', 1),
     -1,
     ('tensor_jit_index', 12),
     ('tensor_jit_index', 11),
     ('tensor_jit_index', 10),
     ('tensor_jit_index', 9),
     ('tensor_jit_index', 8),
     ('tensor_jit_index', 7),
     ('tensor_jit_index', 6),
     0.1,
     1e-05,
     None,
     2,
     1,
     False,
     0,
     116,
     True,
     ('tensor_jit_index', 5),
     3,
     ('tensor_jit_index', 1),
     -1,
     ('tensor_jit_index', 16),
     ('tensor_jit_index', 15),
     ('tensor_jit_index', 14),
     0.1,
     1e-05,
     None,
     2,
     1,
     False,
     0,
     232,
     True,
     ('tensor_jit_index', 13),
     3,
     ('tensor_jit_index', 1),
     -1,
     0.1,
     1e-05,
     None,
     1,
     0,
     False,
     True,
     2,
     3,
     False,
     None,
     1)),