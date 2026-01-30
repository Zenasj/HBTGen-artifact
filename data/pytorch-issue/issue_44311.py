import torch

my_tensor_cpu = torch.arange(3*3*3).view(3,3,3)

#normal indexing
print(my_tensor_cpu[:,[2],2])

#yields 
# tensor([[ 8],
#         [17],
#         [26]])

#indexing with error
print(my_tensor_cpu[:,[10],2])

#yields
# IndexError: index 10 is out of bounds for dimension 0 with size 3

my_tensor_gpu = torch.arange(3*3*3).view(3,3,3).to('cuda')

# normal indexing
print(my_tensor_gpu[:,[2],2])

# yields 
# tensor([[ 8],
#         [17],
#         [26]], device='cuda:0')

# indexing with error
print(my_tensor_gpu[:,[10],2])

# yields
# tensor([[0],
#         [0],
#         [0]], device='cuda:0')