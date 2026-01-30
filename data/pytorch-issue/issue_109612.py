import torch.nn as nn

python
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
from tqdm import tqdm

model_.eval()
qconfig = get_default_qconfig("qnnpack")
qconfig_mapping = QConfigMapping().set_global(qconfig)

qconfig_dict = {
    "": qconfig,  
    "module_name": [("output_layer", None)],
}

example_inputs = torch.rand(1, 3, 112, 112).cpu()  
prepared_model = prepare_fx(model_, qconfig_dict, example_inputs)
calibrate(prepared_model, data, steps=200) 
quantized_model = convert_fx(prepared_model)

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from tqdm import tqdm

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1),
            torch.nn.BatchNorm2d(3),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Dropout(0),
            torch.nn.Flatten(),
            torch.nn.Linear(300, 3),
            torch.nn.BatchNorm1d(3),
        )

    def forward(self, x):
        x = self.res_layer(x)
        return x

x = torch.randn(10, 3, 10, 10)
model_ = Net()
model_.res_layer[0].weight.data = torch.ones(3,3,1,1) 
model_(x)
model_(x*.5)

model_.eval()

y_normal = model_(x)
y_without_bn = model_.res_layer[0](x)

print("diff with/without bn", (y_normal-y_without_bn).abs().sum())

qconfig = get_default_qconfig("qnnpack")
qconfig_mapping = QConfigMapping().set_global(qconfig)

qconfig_dict = {
    "": qconfig,
    "module_name": [("output_layer", None)],
}
# [(x*a+b-m)/s]
# x*a/s + (b-m)/s

print("expected folded weight", (model_.res_layer[0].weight/model_.res_layer[1].running_var**(1/2))[0].flatten())
prepared_model = prepare_fx(model_, qconfig_dict, x)
print("weight after prepare", getattr(prepared_model.res_layer, "0").weight.flatten())
y_after = prepared_model(x)

print("diff before/after prepare", (y_normal - y_after).abs().sum())

quantized_model = convert_fx(prepared_model)