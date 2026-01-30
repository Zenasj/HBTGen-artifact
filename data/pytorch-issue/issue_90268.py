import torch
import torchvision

input_ = torch.randn(1, 3, 224, 224, requires_grad=False)
model = torchvision.models.get_model('swin_v2_t', weights="DEFAULT")
torch.onnx.export(model, input_, "swin_v2_t.onnx", verbose=True)