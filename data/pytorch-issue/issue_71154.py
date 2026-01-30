import torch.nn as nn

import torch
from matplotlib.pyplot import imshow
# the following ex_z is an input
ex_z = torch.zeros((1,3,1024,1024))
for i in range(100,ex_z.size(-1),800):
    ex_z[:,:,i:i+30,:] = 1
    ex_z[:,:,:,i:i+30] = 1
imshow(ex_z[0,0], interpolation='nearest')

out_channel=1
model = torch.nn.Conv2d(3, out_channel, 3, stride=1, padding=1, bias=False)
model.eval()
out_z = model(ex_z)

out_z -=out_z.min()
out_z /= out_z.max()
n=1
print(out_z.size())      # torch.Size([1, 1, 1024, 1024])
feat1 = out_z.squeeze().detach().numpy()
imshow(feat1, interpolation='nearest')

slice = out_z[0,0,0,:]
left_col = slice[90:140]
right_col = slice[890:940]
print(left_col)
print(right_col)
print('equal:', torch.allclose(left_col, right_col))