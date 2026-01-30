from torch.nn import CrossEntropyLoss
import torch
torch.manual_seed(42)

loss_fct = CrossEntropyLoss(reduction='none')
loss_fct_mean = CrossEntropyLoss(reduction='mean')

logits = torch.randn(1,5,10)
labels = torch.tensor([[-100,-100,-100,-100,-100]]) 

nll_loss = loss_fct(
    logits.view(-1, logits.size(-1)), labels.view(-1)
)

nll_loss_mean = loss_fct_mean(
    logits.view(-1, logits.size(-1)), labels.view(-1)
)

print(f"nll_loss is {nll_loss}")
print(f"nll_loss_mean is {nll_loss_mean}")

# nll_loss is tensor([0., 0., 0., 0., 0.])
# nll_loss_mean is nan

input = torch.randn(2,5,2, requires_grad=False)
w = torch.randn(2,2,requires_grad=True)
logits = torch.matmul(input,w)
target = torch.tensor([[-100,-100,-100,-100,-100],[-100,-100,-100,-100,-100]])
output = loss_fct_mean(logits.view(-1, input.size(-1)), target.view(-1))
print(output)
output.backward()
print(w.grad)

# output: tensor(nan, grad_fn=<NllLossBackward0>)
# output: tensor([[0., 0.],
#               [0., 0.]])