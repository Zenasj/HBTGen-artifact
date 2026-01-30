import torch
import torch.nn as nn
import torch.nn.functional as F

def precision(output, target, top_k=(1,)):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, top_k_predicted_classes = output.topk(max_k, dim=-1, largest=True, sorted=True)
        top_k_predicted_classes_t = top_k_predicted_classes.t()
        correct = top_k_predicted_classes_t == target.view(1, -1).expand_as(top_k_predicted_classes_t)

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdims=True)
            res.append((correct_k / batch_size))

        return res

device_mps = torch.device("mps")
device_cpu = torch.device("cpu")
# device_cuda = torch.device("cuda")

output = torch.tensor([
    [-0.0855, -0.1493,  0.1863,  0.3727,  0.2557, -0.3155, -0.0390,  0.2291, -0.1315,  0.5153],
    [-0.2017,  0.3050,  0.3284, -0.0479, -0.0203,  0.4262,  0.3589, -0.1662, -0.0310, -0.5451],
    [ 0.1809,  0.1366,  0.0176, -0.0035, -0.1054, -0.1738,  0.2088, -0.1329, -0.2929, -0.1928],
    [ 0.1760,  0.0382,  0.0762, -0.5019,  0.3571,  0.2943, -0.4008,  0.1386, -0.1081, -0.3811],
    [-0.3708, -0.0578,  0.1646,  0.2789,  0.4265,  0.0539,  0.3712, -0.2590, 0.0785, -0.2830],
    [-0.0354, -1.5403,  0.1378,  1.5788, -1.7130, -1.7471,  1.2042, -0.8079, 0.3714,  0.0538]
])
target = torch.tensor([9, 4, 0, 2, 7, 1])

print("Output on MPS: ", precision(output.to(device_mps), target.to(device_mps), top_k=(1, 5)))
print("Output on CPU: ", precision(output.to(device_cpu), target.to(device_cpu), top_k=(1, 5)))
# print("Output on CUDA: ", precision(output.to(device_cuda), target.to(device_cuda), top_k=(1, 5)))