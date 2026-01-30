import torch.nn as nn

py
import torch
import torchvision


# Needed since lists and dicts are not supported as output types.
# We simply put them inside a tuple.
class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        out = self.model(input)
        return out[0]["boxes"], out[0]["scores"], out[0]["labels"], out[0]["masks"]

print('Torch', torch.__version__, 'TorchVision', torchvision.__version__)

# An instance of your model.
model = TraceWrapper(torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True))

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 320, 320)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.eval()

# Save the TorchScript model
traced_script_module.save("maskrcnn-resnet50-fpn.pt")