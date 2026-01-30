import torch

tensor_cpu = torch.LongTensor([[0], [1]]).expand(2, 4)
mask_cpu = torch.BoolTensor(
    [[False,  True, False, False],
     [False, False, False, False]]
    )
tensor_cpu.masked_fill_(mask_cpu, 3)
print(tensor_cpu)

tensor_cpu = torch.LongTensor([[0], [1]]).expand(2, 4)
masked_tensor_cpu = tensor_cpu.masked_fill(mask_cpu, 3)
print(masked_tensor_cpu)

tensor_cuda = torch.LongTensor([[0], [1]]).expand(2, 4).to('cuda')
mask_cuda = torch.BoolTensor(
    [[False,  True, False, False],
     [False, False, False, False]]
    ).to('cuda')
masked_tensor_cuda = tensor_cuda.masked_fill(mask_cuda, 3)
print(masked_tensor_cuda)                   # out-of-place version
tensor_cuda.masked_fill_(mask_cuda, 3)
print(tensor_cuda)                          # in-place version

other_cpu = torch.LongTensor(
    [[0, 0, 0, 0],
     [1, 1, 1, 1]]
    )
print(other_cpu.masked_fill_(mask_cpu, 3))

tensor_cuda = torch.LongTensor([[0], [1]]).to('cuda').expand(2, 4)
tensor_cuda.masked_fill_(mask_cuda, 3)
print(tensor_cuda)