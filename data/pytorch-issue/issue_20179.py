import torch
torch.manual_seed(0) 
torch.cuda.set_device(0)

cnt = 0 
while True:
    randns = torch.empty((10000000,), device=torch.cuda.current_device()).exponential_() 
    #randns = torch.empty((10000000,)).exponential_() 
    gumbel = -randns.log() 

    cnt += 1
    idxes = torch.isinf(gumbel)
    if idxes.any():
        _, idx = torch.max(idxes, 0)
        print('{} is sampled in the {}-th entry in the {}-th sampling'.format(randns[idx], idx, cnt))
        break
    else:
        print('{}'.format(cnt))

import torch
import torch.nn as nn

class TEST(nn.Module):
  def __init__(self):
    super(TEST, self).__init__()
    self.register_parameter('weight', nn.Parameter(torch.Tensor(100, 9)))
    nn.init.normal_(self.weight, 0, 0.01)

  def forward(self, inputs):
    logits  = self.weight
    nn.init.normal_(logits, 0, 0.01)
    gumbels = -torch.empty_like(logits).exponential_().log()
    new_logits = (logits + gumbels) / 0.5
    probs = nn.functional.softmax(new_logits, dim=1).cpu()
    selected_index = torch.multinomial(probs + 1e-7, 2, False).to(logits.device)
    
test = TEST()
test = nn.DataParallel(test).cuda()
inputs = torch.Tensor(4, 5)

for i in range(100000):
  test(inputs)