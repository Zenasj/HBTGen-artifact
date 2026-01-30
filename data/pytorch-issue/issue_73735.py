import torch.nn as nn

import torch
import torchvision
from faster_rcnn.frcnn import FRCNN

# An instance of your model.
frcnn = FRCNN()
model = frcnn.net

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")