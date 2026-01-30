import torch.nn as nn

import torch
from torch import fx
from torch.fx import symbolic_trace
from torch.quantization import get_default_qat_qconfig
import torch.quantization.quantize_fx as quantize_fx


# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward_train(self, x, gt_label, **kwargs):
        y = self.linear(x + self.param).clamp(min=0.0, max=1.0)
        # gt = kwargs['gt_label']
        gt = gt_label
        diff = y - gt
        return torch.mean(diff)
     
    def forward(self, x, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(x, **kwargs)
        else:
            return self.linear(x + self.param).clamp(min=0.1, max=1.0)

net = MyModule()

import copy
net_qat = copy.deepcopy(net)
net_qat.train()
qconfig = get_default_qat_qconfig('qnnpack')
qconfig_dict = {"": qconfig}
# prepare
model_prepared = quantize_fx.prepare_qat_fx(net_qat,
                                            qconfig_dict)

model = model_prepared
model.graph.print_tabular()
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()
loss_func = torch.nn.L1Loss()

for i in range(100):
    x = torch.rand(3, 4)
    y = torch.rand(3, 5)
    loss = model(x, True, img_metas=None, gt_label=y)
    # print(out.shape, y.shape)
    # loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    print(loss)
    optimizer.step()

model_prepared = quantize_fx.prepare_qat_fx(net_qat,
                                            qconfig_dict,
                                            concrete_args={
                                                'return_loss': True,
                                                'img_metas': torch.fx.PH,
                                                'gt_label': torch.fx.PH
                                            })