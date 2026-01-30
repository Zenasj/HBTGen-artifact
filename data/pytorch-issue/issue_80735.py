import time
import torch as th

N = 100
X = th.randn(153531, 4, 4, device="cuda")

X.inverse()

th.cuda.synchronize()
start = time.time()

for _ in range(N):
    X.inverse()

th.cuda.synchronize()
end = time.time()

print("Time per inverse (ms):", 1000 * (end - start) / N)
print("PyTorch Version:", th.__version__)

import time
import torch as th
import torch
torch.manual_seed(0)

N = 100
X = th.randn(153531, 4, 4, device="cuda")

# func_cls=torch.cholesky_inverse # 39/2
# func_cls=torch.pinverse# 190/162
# func_cls=torch.Tensor.inverse # 42/0.26
func_cls=torch.Tensor.cholesky_inverse# 39/2

func_cls(X)


th.cuda.synchronize()
start = time.time()


for _ in range(N):
    func_cls(X)


th.cuda.synchronize()
end = time.time()

print("Time per inverse (ms):", 1000 * (end - start) / N)
'''.
torch.cholesky_inverse: 
    1.12.0: 39.43123817443848 / 1.13.0: 2.6720738410949707
torch.pinverse: 
    1.12.0: 190.62803745269775 / 1.13.0: 162.25743532180786
torch.cholesky_inverse: 
    1.12.0: 42.09115982055664 / 1.13.0: 0.2670574188232422
torch.cholesky_inverse: 
    1.12.0: 39.01032209396362 / 1.13.0: 2.949252128601074
'''