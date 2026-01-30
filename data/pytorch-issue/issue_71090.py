import torch
a = torch.randn([1,9,2], device="cuda")
b = torch.randn([1,2,9,2], device="cuda")
c = torch.einsum("bsl,bcsl->bcl", a, b).reshape(1, 2, 1, 2)
assert c.is_contiguous() == False
d = c.contiguous()
assert d.is_contiguous() == True

torch.manual_seed(0)
print(torch.randn_like(c))

torch.manual_seed(0)
print(torch.randn_like(d))

torch.manual_seed(0)
print(torch.randn_like(c, memory_format=torch.contiguous_format))

torch.manual_seed(0)
print(torch.randn_like(d, memory_format=torch.contiguous_format))

torch.manual_seed(0)
print(torch.randn_like(c).contiguous())
torch.manual_seed(0)
print(torch.randn_like(c.contiguous()))