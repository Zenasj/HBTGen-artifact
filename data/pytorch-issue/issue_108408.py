import torch

array = torch.empty(10, 10, dtype=torch.float32, device="cuda")
print(array.data_ptr())
array_2 = torch.asarray(array, copy=True, device="cuda")
print(array_2.data_ptr())

array[0,0] = 0
array_2[0,0] = 10
print(array[0,0])