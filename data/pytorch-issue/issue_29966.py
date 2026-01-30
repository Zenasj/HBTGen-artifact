import torch

max = torch.tensor([3])
if USE_CUDA: max = max.cuda()
max_embedding = self.max_embedding(max) # dim of max_embedding: 1*5

item_dict = {}
for item in item_list:
    item = torch.tensor(item)
    if USE_CUDA: item = item.cuda()
    item_embedding = self.item_embedding(item) # dim of item_embedding: 1*20

embedded = torch.cat((max_embedding, item_embedding), 1)