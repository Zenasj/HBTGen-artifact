import torch
import random

import numpy as np
np.random.seed(1335087076)
# device = "cpu"
tensor_0 = torch.from_numpy(np.random.uniform(0.0, 1.0, size=[3, 3, 2, 2])).to(torch.float32).requires_grad_(False)


def basic(tensor_0):
    # export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0
    tensor_3 = tensor_0.unsqueeze_(1)
    return tensor_3

dynamo_res = torch.compile(basic, backend="inductor")(tensor_0.clone())

cpu_res = basic(tensor_0.clone())

print("res: ", dynamo_res.shape, cpu_res.shape)