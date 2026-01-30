import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
example = torch.randn(1, 3, 1000, 1000)
scripted_model = torch.jit.script(model)
graph, params = torch._C._jit_pass_lower_graph(scripted_model.forward.graph, scripted_model._c)