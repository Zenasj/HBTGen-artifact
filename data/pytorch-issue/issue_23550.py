# test1.py
import torch
import torch.nn as nn
import random

torch.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv = nn.Conv3d(3,64,(1,7,7),stride=(1,2,2),padding=(0,3,3),bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1,3,3),stride=(1,2,2))
    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))
        #return self.relu(self.conv(x))

a = torch.rand(2,3,8,64,64)
m = model()
a = a.to("cuda:0")
m.to("cuda:0")
b = m(a)
b = b.view(2, -1)
fc = nn.Linear(b.size(1), 80)
fc.to("cuda:0")
pred = fc(b)
label = torch.randint(2, (2,80), dtype=torch.float32, device="cuda:0")
bceloss = nn.BCEWithLogitsLoss()
loss = bceloss(pred, label)
print(loss.item())

loss.backward()
tensor_dict = dict(a=a,b=b,pred=pred,label=label,loss=loss)
torch.save(tensor_dict, "tensor1.pth")
model_dict = dict(conv_w=m.conv.weight, fc_w=fc.weight, fc_b=fc.bias)
torch.save(model_dict, "model1.pth")
grad_dict = dict(conv_w_grad=m.conv.weight.grad, fc_w_grad = fc.weight.grad, fc_b_grad = fc.bias.grad)
torch.save(grad_dict, "grad1.pth")

# test2.py
import torch
import torch.nn as nn
import random

torch.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv = nn.Conv3d(3,64,(1,7,7),stride=(1,2,2),padding=(0,3,3),bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1,3,3),stride=(1,2,2))
    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))
        #return self.relu(self.conv(x))

a = torch.rand(2,3,8,64,64)
m = model()
a = a.to("cuda:0")
m.to("cuda:0")
b = m(a)
b = b.view(2, -1)
fc = nn.Linear(b.size(1), 80)
fc.to("cuda:0")
pred = fc(b)
label = torch.randint(2, (2,80), dtype=torch.float32, device="cuda:0")
bceloss = nn.BCEWithLogitsLoss()
loss = bceloss(pred, label)
print(loss.item())

loss.backward()
tensor_dict = dict(a=a,b=b,pred=pred,label=label,loss=loss)
torch.save(tensor_dict, "tensor2.pth")
model_dict = dict(conv_w=m.conv.weight, fc_w=fc.weight, fc_b=fc.bias)
torch.save(model_dict, "model2.pth")
grad_dict = dict(conv_w_grad=m.conv.weight.grad, fc_w_grad = fc.weight.grad, fc_b_grad = fc.bias.grad)
torch.save(grad_dict, "grad2.pth")

# cmp.py
import torch

def relative_diff(a,b):
    rel = abs(a-b)/abs(a)
    zero_mask = (a==0)&(b==0)
    rel[zero_mask] = 0
    rel[torch.isnan(rel)]=1000
    return rel

def cmp(path1, path2):
    cmp1 = torch.load(path1, map_location="cpu")
    cmp2 = torch.load(path2, map_location="cpu")
    for key in cmp1:
        cmp11, cmp22 = cmp1[key], cmp2[key]
        diff = relative_diff(cmp11, cmp22)
        print("max of {} diff: {}".format(key, diff.max().item()))
        print("sum of {} diff: {}".format(key, diff.sum().item()))

tensor1 = "tensor1.pth"
tensor2 = "tensor2.pth"
print("Compare between tensors:")
cmp(tensor1, tensor2)

model1 = "model1.pth"
model2 = "model2.pth"
print("Compare between model weights:")
cmp(model1, model2)

grad1 = "grad1.pth"
grad2 = "grad2.pth"
print("Compare between gradient:")
cmp(grad1, grad2)