import torch.nn as nn

import math
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,2,3)
        self.conv2 = torch.nn.Conv2d(1,2,4)
        self.new_parameters = self.conv1.parameters()
        self.pretrained_parameters = self.conv2.parameters()

def test_plateau_alone():
    m = Model()
    opt = torch.optim.SGD([
        {"params":m.new_parameters, "lr":1.0},
        {"params":m.pretrained_parameters, 'lr':1.0}
        ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
        factor=.1, patience=0, verbose=True, mode='min')
    opt.step(); scheduler.step(1);
    assert math.isclose(opt.param_groups[0]["lr"], 1.0)
    opt.step(); scheduler.step(1);
    assert math.isclose(opt.param_groups[0]["lr"], 0.1)
    print("This is fine")

def test_plateau_with_friend():
    m = Model()
    opt = torch.optim.SGD([
        {"params":m.new_parameters, "lr":1.0},
        {"params":m.pretrained_parameters, 'lr':1.0}
        ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
        factor=.1, patience=0, verbose=True, mode='min')
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(opt, [lambda _: 1, lambda _: 1])
    opt.step(); scheduler.step(1); scheduler2.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0)
    opt.step(); scheduler.step(1); scheduler2.step()
    print("Got lr: " + str(opt.param_groups[0]["lr"]))
    assert math.isclose(opt.param_groups[0]["lr"], 0.1)

print(f"TORCH VERSION: {torch.__version__}")
test_plateau_alone()
test_plateau_with_friend()