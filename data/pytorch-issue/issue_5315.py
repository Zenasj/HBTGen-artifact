import torch.nn as nn
from torch.autograd import Variable
import torch.onnx
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dummy = Variable(torch.randn(10, 3, 224, 224, device='cuda'))

model = torchvision.models.resnet50(pretrained=True)

model.to(device)

torch.onnx.export(model, dummy, "resnet50.pb")

state_dict = torch.load('/path/to/your/.pth/model')
model.load_state_dict(state_dict)
model.eval()
dummy_input = Variable(torch.randn(B, C, H, W))
torch.onnx.export(model.module, dummy_input, '/path/to/output/onnx/model', export_params = True)