import torch

values = [
    [0.8909, 0.8909, 0.8909],
    [0.8909, 0.8909, 0.8909],
    [0.8909, 0.8909, 0.8909]
]
singleTensor = torch.tensor(values, dtype=torch.float64)
multiTensor = torch.tensor([values, values, values], dtype=torch.float64)

print("CPU single: %s, multi: %s" % (torch.det(singleTensor), torch.det(multiTensor)))
print("GPU single: %s, multi: %s" % (torch.det(singleTensor.cuda()), torch.det(multiTensor.cuda())))