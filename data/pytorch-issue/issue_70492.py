import torch.nn as nn

results = dict()
import torch
import time
arg_1_tensor = torch.rand([4, 141, 768], dtype=torch.float16)
arg_2_tensor = torch.rand([768, 768], dtype=torch.float16)
arg_3_tensor = torch.rand([768], dtype=torch.float16)

arg_1 = arg_1_tensor.clone().type(torch.float32)
arg_2 = arg_2_tensor.clone().type(torch.float32)
arg_3 = arg_3_tensor.clone().type(torch.float32)
start = time.time()
torch.nn.functional.linear(arg_1,arg_2,arg_3,)
results["time_high"] = time.time() - start

arg_1 = arg_1_tensor.clone()
arg_2 = arg_2_tensor.clone()
arg_3 = arg_3_tensor.clone()
start = time.time()
torch.nn.functional.linear(arg_1,arg_2,arg_3,)
results["time_low"] = time.time() - start

print(results)
# {'time_high': 0.06454730033874512, 'time_low': 8.320769309997559}