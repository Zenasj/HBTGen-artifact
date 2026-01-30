import torch
import torch.nn as nn

conv_no_warn = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=0).eval().cuda()
conv_warn = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=0).eval().cuda()
x = torch.rand((1, 8, 546, 392)).cuda()

with torch.inference_mode(), torch.autocast(device_type="cuda"):
    # No warning
    conv_no_warn(x)
    # UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED
    conv_warn(x)

import torch
import torch.nn as nn

conv_warn = nn.Conv2d(768, 96, kernel_size=(1, 1), stride=(1, 1)).eval().cuda()
x = torch.rand((1, 768, 39, 28)).cuda()

with torch.inference_mode(), torch.autocast(device_type="cuda"):
    # UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED
    conv_warn(x)