import torch as t
import numpy as np

Xt = t.linalg.solve(t.Tensor(A),t.Tensor(B))
print('torch ',t.__version__,Xt.min(),Xt.max(),(t.Tensor(A) @ Xt - t.Tensor(B)).pow(2).sum())
Xt32 = t.linalg.solve(t.Tensor(A).to(t.float32),t.Tensor(B).to(t.float32))
print('torch 32',t.__version__,Xt32.min(),Xt32.max(),(t.Tensor(A).to(t.float32) @ Xt32 - t.Tensor(B).to(t.float32)).pow(2).sum())
Xt64 = t.linalg.solve(t.Tensor(A).to(t.float64),t.Tensor(B).to(t.float64))
print('torch 64',t.__version__,Xt64.min(),Xt64.max(),(t.Tensor(A).to(t.float64) @ Xt64 - t.Tensor(B).to(t.float64)).pow(2).sum())

Xn = np.linalg.solve(np.array(A),np.array(B))
print('numpy',np.__version__,Xn.min(),Xn.max(),((np.array(A) @ Xn - np.array(B))**2).sum())
Xn32 = np.linalg.solve(np.array(A,dtype=np.float32),np.array(B,dtype=np.float32))
print('numpy 32',np.__version__,Xn32.min(),Xn32.max(),((np.array(A,dtype=np.float32) @ Xn32 - np.array(B,dtype=np.float32))**2).sum())
Xn64 = np.linalg.solve(np.array(A,dtype=np.float64),np.array(B,dtype=np.float64))
print('numpy 64',np.__version__,Xn64.min(),Xn64.max(),((np.array(A,dtype=np.float64) @ Xn64 - np.array(B,dtype=np.float64))**2).sum())

Xt32cu = t.linalg.solve(t.Tensor(A).to(t.float32).cuda(),t.Tensor(B).to(t.float32).cuda())
print('torch 32 cuda',t.__version__,Xt32cu.min(),Xt32cu.max(),(t.Tensor(A).to(t.float32).cuda() @ Xt32cu - t.Tensor(B).to(t.float32).cuda()).pow(2).sum())
Xt64cu = t.linalg.solve(t.Tensor(A).to(t.float64).cuda(),t.Tensor(B).to(t.float64).cuda())
print('torch 64 cuda',t.__version__,Xt64cu.min(),Xt64cu.max(),(t.Tensor(A).to(t.float64).cuda() @ Xt64cu - t.Tensor(B).to(t.float64).cuda()).pow(2).sum())