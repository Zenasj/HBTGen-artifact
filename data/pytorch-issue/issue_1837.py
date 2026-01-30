import torch
import numpy as np
y =[1, 2, 3, 5, 9, 1]
print("numpy=",np.median(y))
print(sorted([1, 2, 3, 5, 9, 1]))
yt = torch.tensor(y,dtype=torch.float32)
ymax = torch.tensor([yt.max()])
print("torch=",yt.median())
print("torch_fixed=",(torch.cat((yt,ymax)).median()+yt.median())/2.)