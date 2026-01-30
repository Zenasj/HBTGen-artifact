import torch
bool_list = [True, True, False, False]

tensor1 = torch.tensor(bool_list)
tensor2 = torch.Tensor(bool_list)

print(tensor1)
# tensor([ True,  True, False, False])

print(tensor2)
# tensor([1., 1., 0., 0.])