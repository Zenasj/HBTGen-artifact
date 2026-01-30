import torch

t1 = torch.ones((3,2,0)).cuda(0)
t2 = torch.ones((3,0,3)).cuda(0)
res_bmm = torch.bmm(t1, t2)
res_matmul = torch.matmul(t1, t2)

print("===== [ res_bmm ] =====")
print(res_bmm)
print()
print("===== [ res_matmul ] =====")
print(res_matmul)