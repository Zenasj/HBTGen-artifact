import torch
mv = torch.arange(5,device=torch.device('mps'),dtype=torch.float32)
cv = torch.zeros(5,dtype=torch.float32)
for i in range(5):
    cv[i] = mv[i]
print(f'mv={mv.cpu().numpy()}')
print(f'cv={cv.numpy()}')

import torch
mv = torch.arange(5,device=torch.device('mps'),dtype=torch.int32)
cv = torch.zeros(5,dtype=torch.float32)
for i in range(5):
    cv[i] = mv[i]
print(f'mv={mv.cpu().numpy()}')
print(f'cv={cv.numpy()}')