import torch
arg_1 = torch.tensor([0,1,2,3,4])
arg_2 = 6
res1 = torch.kthvalue(arg_1,arg_2,)
res3 = torch.kthvalue(arg_1,arg_2,)
res2 = arg_1.kthvalue(arg_2,)
print(res1)
print(res2)
print(res3)

torch.return_types.kthvalue(
values=tensor(65),
indices=tensor(65))
torch.return_types.kthvalue(
values=tensor(8315178049699258475),
indices=tensor(81))
torch.return_types.kthvalue(values=tensor(4423495177514940772),indices=tensor(4424065875907452991))