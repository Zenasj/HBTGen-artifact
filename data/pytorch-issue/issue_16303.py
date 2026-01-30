import torch
 
n = int(1e7)
batch_sz = int(n / 10)

x = torch.tensor([1, 2], requires_grad=True, dtype=torch.float32, device="cpu")
# x = torch.tensor([1, 2], dtype=torch.float32, device="cpu")

# some huge tensor not fitting into the gpu memory
z = torch.randn(2, n, dtype=torch.float32, device="cpu")
 
batch_lst = []
 
for i in range(0, n, batch_sz):
    batch_lst.append(
        # do some heavy-weight computations on gpu
        torch.max((
            x.view([2, 1]).cuda() * \
            z[:, i:(i + batch_sz)].cuda()
            )**2,
            dim=0)[0].cpu() # and move back to cpu
    )
    print("allocated: %.2fG" % (torch.cuda.memory_allocated() / 1e9),
          "cached: %.2fG" % (torch.cuda.memory_cached() / 1e9))
 
# aggregate batches on cpu
f = torch.mean(torch.cat(batch_lst))
print("f:", f, f.device)
print("allocated: %.2fG" % (torch.cuda.memory_allocated() / 1e9),
      "cached: %.2fG" % (torch.cuda.memory_cached() / 1e9))
 
# gradient on cpu
g = torch.autograd.grad(f, x, retain_graph=True, create_graph=True)[0]
print("g:", g, g.device)
print("allocated: %.2fG" % (torch.cuda.memory_allocated() / 1e9),
      "cached: %.2fG" % (torch.cuda.memory_cached() / 1e9))

# 1st row of hessian on cpu
h1 = torch.autograd.grad(g[0], x, retain_graph=True, create_graph=True)[0]
print("h1:", h1, h1.device)
print("allocated: %.2fG" % (torch.cuda.memory_allocated() / 1e9),
      "cached: %.2fG" % (torch.cuda.memory_cached() / 1e9))

# 2nd row of hessian on cpu
h2 = torch.autograd.grad(g[1], x, retain_graph=True, create_graph=True)[0]
print("h2:", h2, h2.device)
print("allocated: %.2fG" % (torch.cuda.memory_allocated() / 1e9),
      "cached: %.2fG" % (torch.cuda.memory_cached() / 1e9))