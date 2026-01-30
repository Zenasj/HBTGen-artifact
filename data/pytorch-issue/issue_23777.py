import torch
import torch.nn as nn
import torch.optim as optim

def seed_torch(seed):
    import random
    random.seed(seed)
    
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_torch(2019)

class Model(nn.Module):
    def __init__(self, vectors):
        super(Model, self).__init__()
        self.emb = nn.Embedding.from_pretrained(
            vectors, freeze=False,
            padding_idx=0
        )

        # dummy embedding layers, just initialized and not used
        self.emb2 = nn.Embedding.from_pretrained(
            vectors, freeze=False,
            padding_idx=0
        )
        self.linear = nn.Linear(128, 2)
        
    def forward(self, s):
        s = self.emb(s)
        s = self.linear(s)
        s = s.sum(dim=1)
        return s
    
vectors = torch.randn(1000, 128)
# fake samples
train_s = torch.randint(1, 1000, size=(100, 30)).to('cuda')
train_y = torch.ones(100).to(dtype=torch.long).to('cuda')

# train
model = Model(vectors).to('cuda')
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model.train()
model.zero_grad()
out = model(train_s)
loss = criterion(out, train_y)
loss.backward()
optimizer.step()

torch.save(model.state_dict(), 'tmp.pt')

# inference, model init -> to cuda -> load state dict
m1 = Model(vectors).to('cuda')
m1.load_state_dict(torch.load('tmp.pt'))
m1.eval()
with torch.no_grad():
    o1 = m1(train_s.to('cuda'))
    
# inference, model init -> load state dit -> to cuda
m2 = Model(vectors)
m2.load_state_dict(torch.load('tmp.pt'))
m2 = m2.to('cuda')
m2.eval()
with torch.no_grad():
    o2 = m2(train_s.to('cuda'))
    
# inference on cpu
m3 = Model(vectors)
m3.load_state_dict(torch.load('tmp.pt'))
m3.eval()
with torch.no_grad():
    o3 = m3(train_s.to('cpu'))
    
assert torch.allclose(o1, o2)
assert torch.allclose(o1.cpu(), o3)
assert torch.allclose(o2.cpu(), o3)