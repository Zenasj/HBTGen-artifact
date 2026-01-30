python
import torch

device = "cuda:0"
device = "cpu"
data_store = torch.zeros([5,3]).to(device)
data_store_cp = torch.zeros([5,3]).to(device)
index = torch.randint(0,4,[100]).to(device)
data = torch.randn([index.size(0),3]).to(device)
data_store[index.long()]=data
data_store_cp[index.long()] = data
print(f"equal: {torch.equal(data_store, data_store_cp)}")

device = "cpu"

device = "cuda:0"