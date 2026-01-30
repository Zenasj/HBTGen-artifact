import torch

param = torch.zeros(5)
param2 = torch.zeros(5, 10)

tensor_list = [param2]
print(param in tensor_list)  # error - tried to broadcast due to equality

param = torch.zeros(5)
param2 = torch.zeros(5)

tensor_list = [param2]
print(param in tensor_list)  # Fails

param = torch.zeros(5)

tensor_list = [param]
print(param in tensor_list)  # True, due to hash equality

param = torch.zeros(5)
param2 = torch.zeros(5, 10)

tensor_list = set({param2})
print(param in tensor_list)   # False, as we never had to go to equality check, due to first hashing on other items