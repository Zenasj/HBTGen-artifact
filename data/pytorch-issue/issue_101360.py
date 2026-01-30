import torch

tensor_list = []
for i in range(10**7):
    tensor_list.append(torch.zeros(8))
big_tensor = torch.stack(tensor_list)