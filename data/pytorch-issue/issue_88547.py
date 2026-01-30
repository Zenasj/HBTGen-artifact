py
import torch

cuda_output = torch.bitwise_right_shift(torch.tensor(1).cuda(), 1)
print(cuda_output)

cpu_output = torch.bitwise_right_shift(torch.tensor(1), 64)
print(cpu_output)

tensor(0, device='cuda:0')
tensor(1)

py
for i in range(1, 1000):
    cpu_output = torch.bitwise_right_shift(torch.tensor(1), i)
    if cpu_output != 0:
        print(i, cpu_output)

py
import torch

a = torch.tensor(1)
cpu_output = torch.bitwise_left_shift(a, -1)
cuda_output = torch.bitwise_left_shift(a.cuda(), -1)

print(cpu_output)
print(cuda_output)

tensor(-9223372036854775808)
tensor(0, device='cuda:0')