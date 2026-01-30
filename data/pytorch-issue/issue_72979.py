import io
import torch
import torchvision as tv

model = tv.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
with io.BytesIO() as f:
    torch.onnx.export(model, x, f, opset_version=11)