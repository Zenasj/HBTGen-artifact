import torch, pickle

tensor = torch.rand([1,2,3],dtype=torch.float32).to(torch.complex32)
with open('test.pkl', 'wb') as f : pickle.dump(tensor, f)

import torch
tensor = torch.rand([1,2,3],dtype=torch.float32).to(torch.complex32)
torch.save(tensor)