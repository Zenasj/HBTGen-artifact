import torch
import torchvision.models as models
device = torch.device("cuda")
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(device)
comp = torch.compile(options={"trace.enabled": True, "trace.save_real_tensors": True})(models.resnet50(pretrained=True).to(device))
print(comp(input_tensor))