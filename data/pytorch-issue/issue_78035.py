import torchvision

import torch
import numpy as np
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

np.random.seed(1337)
torch.manual_seed(1337)

X_base, y_base = np.random.rand(16, 3, 64, 64).astype(np.float32), np.random.choice(np.arange(10), size=16)
base_model_state_dict = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10).state_dict()
criterion = nn.CrossEntropyLoss()

for i in range(5):
    device = "cpu"
    cpu_model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10)
    cpu_model.load_state_dict(base_model_state_dict)
    cpu_model = cpu_model.to(device)
    cpu_model.train()
    cpu_model.zero_grad()

    loss = criterion(cpu_model(torch.from_numpy(X_base.copy()).to(device)), torch.from_numpy(y_base.copy()).to(device))
    loss.backward()

    device = "mps"
    mps_model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10)
    mps_model.load_state_dict(base_model_state_dict)
    mps_model = mps_model.to(device)
    mps_model.train()
    mps_model.zero_grad()

    loss_mps = criterion(mps_model(torch.from_numpy(X_base.copy()).to(device)), torch.from_numpy(y_base.copy()).to(device))
    loss_mps.backward()

    print(f"[{i+1}/5]: Object location", hex(id(cpu_model)), hex(id(mps_model)))

    for p_cpu, p_mps in zip(cpu_model.parameters(), mps_model.parameters()):
        if p_cpu.requires_grad:
            with torch.no_grad():
                mag_diff = torch.mean(torch.abs(p_cpu.grad.to("cpu") - p_mps.grad.to("cpu")))
                if mag_diff > 1e-3:
                    print(p_cpu.grad.cpu().reshape(-1)[:10])
                    print(p_mps.grad.cpu().reshape(-1)[:10])
                    print("Weight diff", torch.mean(p_cpu.to("cpu") - p_mps.to("cpu")))
                    print("Grad diff", torch.mean(torch.abs(p_cpu.grad.to("cpu") - p_mps.grad.to("cpu"))))
                    print()
                break

    del cpu_model, mps_model

import torch
import numpy as np
import torch.nn as nn
from torchvision.models import alexnet, vgg11, vgg11_bn, mobilenet_v2, resnet18

def seed_everything(seed: int):
    # Ref: https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def inner_loop(i, max_iter, model_fn, X_base, y_base, base_model_state_dict, criterion):
    device = "cpu"
    cpu_model = model_fn(pretrained=False)
    cpu_model.load_state_dict(base_model_state_dict)
    cpu_model = cpu_model.cpu()
    cpu_model.train()
    cpu_model.zero_grad()

    loss = criterion(cpu_model(torch.from_numpy(X_base.copy()).cpu()), torch.from_numpy(y_base.copy()).cpu())
    loss.backward()

    device = "mps"
    mps_model = model_fn(pretrained=False)
    mps_model.load_state_dict(base_model_state_dict)
    mps_model = mps_model.to(device)
    mps_model.train()
    mps_model.zero_grad()

    loss_mps = criterion(mps_model(torch.from_numpy(X_base.copy()).to(device)), torch.from_numpy(y_base.copy()).to(device))
    loss_mps.backward()

    print(f"[{i+1}/{max_iter}]: Object location", hex(id(cpu_model)), hex(id(mps_model)))

    for p_cpu, p_mps in zip(cpu_model.parameters(), mps_model.parameters()):
        if p_cpu.requires_grad:
            with torch.no_grad():
                mag_diff = torch.mean(torch.abs(p_cpu.grad.to("cpu") - p_mps.grad.to("cpu")))
                if mag_diff > 1e-3:
                    print(p_cpu.grad.cpu().reshape(-1)[:10])
                    print(p_mps.grad.cpu().reshape(-1)[:10])
                    print("Weight diff", torch.mean(p_cpu.to("cpu") - p_mps.to("cpu")))
                    print("Grad diff", torch.mean(torch.abs(p_cpu.grad.to("cpu") - p_mps.grad.to("cpu"))))
                    print()
                break

seed_everything(1337)
X_base, y_base = np.random.rand(16, 3, 224, 224).astype(np.float32), np.random.choice(np.arange(1000), size=16)
for model_fn in [alexnet, vgg11, vgg11_bn, resnet18, mobilenet_v2]:
    print(model_fn)
    base_model_state_dict = model_fn(pretrained=False).state_dict()
    criterion = nn.CrossEntropyLoss()
    for i in range(3):
        seed_everything(1337)
        inner_loop(i, 3, model_fn, X_base, y_base, base_model_state_dict, criterion)
    print("-------------------------------")