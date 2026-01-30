import torch
a = torch.rand(size=[10, 212, 500000]).to('cuda:0')
p, i = a.topk(20, sorted=False, dim=-1)
print(i.max())
# prints tensor(2033619248, device='cuda:0')
# while I expect to receive something like tensor(499999, device='cuda:0')

import torch
a = torch.rand(size=[10, 212, 500000])  # .to('cuda:0')
p, i = a.topk(20, sorted=False, dim=-1)
print(i.max())
# correctly prints tensor(499997)

import torch
a = torch.rand(size=[2, 3, 500000]).to('cuda:0')
p, i = a.topk(20, sorted=False, dim=-1)
print(i.max())
# correctly prints tensor(498704, device='cuda:0')

a = torch.rand(size=[10, 212, 500000]).to('cuda:0')
p_list, i_list = [], []
for batch_item in a:
    topk_probs, topk_ids = batch_item.topk(20, sorted=False, dim=-1)
    p_list.append(topk_probs)
    i_list.append(topk_ids)
p_ = torch.stack(p_list, dim=0)
i_ = torch.stack(i_list, dim=0)
print(i_.max())
# should correctly print tensor(499999, device='cuda:0')