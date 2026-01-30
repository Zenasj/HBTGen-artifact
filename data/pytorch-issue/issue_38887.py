import torch.nn as nn

import torch

class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv_stem = torch.nn.Conv2d(
            3, 5, 2, bias=True
        ).to(dtype=torch.float)

        self.bn1 = torch.nn.BatchNorm2d(5)
        self.act1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

    def fuse(self):
        torch.quantization.fuse_modules(
            self, 
            ['conv_stem', 'bn1', 'act1'], 
            inplace=True
        )


def fuse_modules(module):
    module_output = module
    if callable(getattr(module_output, "fuse", None)):
        module_output.fuse()
    for name, child in module.named_children():
        new_child = fuse_modules(child)
        if new_child is not child:
            module_output.add_module(name, new_child)
    return module_output

def create_and_update_model():
    model = ConvModel()
    backend = 'qnnpack'
    model = fuse_modules(model)
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.backends.quantized.engine = backend
    torch.quantization.prepare_qat(model, inplace=True)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    return model

def QAT(model):
    N = 100
    for idx in range(N):
        input_tensor = torch.rand(1, 3, 6, 6)
        model(input_tensor)
    return model

model = create_and_update_model()
model = QAT(model)
torch.quantization.convert(model, inplace=True)

model.eval()
inputs = torch.rand(1, 3, 6, 6)
# Export the model to ONNX
with torch.no_grad():
    with io.BytesIO() as f:
        torch.onnx.export(
            model,
            inputs,
            f,
            opset_version=11,
            verbose=True, 
            export_params=True,
        )