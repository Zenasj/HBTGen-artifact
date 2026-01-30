import torch

int_tensor = torch.tensor([1], dtype=torch.int32)
long_zerodim = torch.tensor(2**31, dtype=torch.int64) 

result = int_tensor.to(torch.int64) + long_zerodim
result_value = result.item()
result_dtype = result.dtype

print(result_value, result_dtype) #2147483649 torch.int64

import torch

int_tensor = torch.tensor([1], dtype=torch.int32)
long_zerodim = torch.tensor([2**31], dtype=torch.int64) 

result = int_tensor + long_zerodim
result_value = result.item()
result_dtype = result.dtype

print(result_value, result_dtype) #2147483649 torch.int64

int_tensor = torch.tensor([0], dtype=torch.int32)
long_zerodim = torch.tensor([2**31], dtype=torch.int64) 
#-2147483648 torch.int32

int_tensor = torch.tensor([0], dtype=torch.int32)
long_zerodim = torch.tensor([2**31], dtype=torch.int64) 
#-2147483647 torch.int32

int_tensor = torch.tensor([2**31-1], dtype=torch.int32)
long_zerodim = torch.tensor(2**31, dtype=torch.int64) 
#-1 torch.int32

int_tensor = torch.tensor([-5], dtype=torch.int32)
long_zerodim = torch.tensor(2**31, dtype=torch.int64) 
#2147483643 torch.int32 torch.int32