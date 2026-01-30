import torch


def new_tensor(tensor, p):
    tensor[:-p] = tensor[p:]


p = 500

tensor = torch.rand(5000, 100).cuda()
tensor_copy = tensor.clone()

new_tensor(tensor, p)
new_tensor(tensor_copy, p)

print('different elements:', (tensor != tensor_copy).nonzero().size(0))