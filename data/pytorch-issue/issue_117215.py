import torch
info=torch.iinfo(torch.uint8)
a = torch.randint(info.min,info.max,(73,11,3,17), dtype=torch.uint8)
b = torch.all(a, dim=0)

c = a.to(torch.bool).all(dim=0)
print(torch.ne(b, c).sum())
# ‘b' and 'c' must be the same

a = torch.randn((73,11,3,17))
info=torch.iinfo(torch.uint8)
out = torch.randint(info.min,info.max,(11,3,17), dtype=torch.uint8)

torch.all(a, dim=0, out=out)
c = (a.to(torch.bool).all(dim=0))

print(torch.ne(out, c).sum())
# ‘b' and 'c' must be the same