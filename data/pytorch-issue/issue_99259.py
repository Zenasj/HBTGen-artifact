import torch
arg_1 = torch.randint(0,512,[2], dtype=torch.int64)
res = torch.repeat_interleave(repeats=arg_1,)
print(res)
# res: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,......

res1 = torch.repeat_interleave(arg_1)
res2 = torch.repeat_interleave(repeats=arg_1)
res3 = torch.repeat_interleave(repeats=arg_1,)