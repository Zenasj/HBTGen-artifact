import torch.nn as nn

import torch as th
def cross_entropy_with_logits(output, target, batch_size):
    probs = th.nn.functional.softmax(output, dim=1)
    loss = -(target * th.log(probs)).mean()
    loss_grad = (probs - target) / (batch_size * target.shape[1])
    return probs, loss, loss_grad

cant_return = th.ones(1)[0]
print(cant_return)
print(cant_return.shape)
this_is_fine = cant_return.reshape(1)
print(this_is_fine)

tensor(1.)
torch.Size([])
tensor([1.])