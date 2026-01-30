for i in range(10000):
    for j in torch.ones(600,30000,device="cuda").multinomial(1000):
        if j.size()!=j.unique().size():
            print('error')

import torch
for i in range(10000):
    torch.manual_seed(i)
    print(i)
    out = torch.ones(600,30000,device="cuda").multinomial(1000)
    for row in out:
        if row.size() != row.unique().size():
            print(row.shape)
            print(row.unique().shape)
            print('error with seed:', i)
            break