import torch
import torch.nn as nn
import torchvision

python
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=5)
model.eval()
model.to(device)

...
# training pipeline
...

q_model = torch.quantization.convert(model)
traced_script_module = torch.jit.trace(q_model, torch.rand(1,3,224,224).to(device))
opt_model = optimize_for_mobile(traced_script_module)
torch.jit.save(opt_model, name)

python
model = torch.jit.load(MYPATH, map_location='cpu')
model.eval()
one = torch.ones([1,3,224,224])
model(one)