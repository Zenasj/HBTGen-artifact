import torchvision

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit(1,0.1)

import torch
import torch.nn as nn
from torchvision.models import resnet34

class OutputHook():
    "Create a hook on `m` with to store the output."
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        self.stored = output
        
    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

torch.set_num_threads(4)
model = resnet34(pretrained=True)
hooks = [OutputHook(m) for m in model.children()]
model.eval()(torch.randn(1, 3, 64, 64))

torch.set_num_threads(4)
model = resnet34(pretrained=True)
hooks = [OutputHook(m) for m in list(model.children())[-1]]
model.eval()(torch.randn(1, 3, 64, 64))