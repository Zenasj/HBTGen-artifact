import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
model = torchvision.models.googlenet(pretrained=True)
ts_model = torch.jit.script(model)
opt_model = optimize_for_mobile(ts_model)

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch._C import MobileOptimizerType
model = torchvision.models.googlenet(pretrained=True)
ts_model = torch.jit.script(model)
optimization_blacklist_no_fold_bn = {MobileOptimizerType.CONV_BN_FUSION}
opt_model = optimize_for_mobile(ts_model, optimization_blacklist_no_fold_bn)