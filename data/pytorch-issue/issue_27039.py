import torch
import torch.nn as nn

batch_size = 1
input_length = 10
num_labels = 3
target_length = 5
targets = torch.randint(1, num_labels, (batch_size * target_length,),
                        device='cuda', dtype=torch.long)
log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='cuda', dtype=torch.float), 2)
log_probs.requires_grad_()

input_lengths = batch_size * [input_length]
target_lengths = batch_size * [target_length]
grad_out = torch.randn(batch_size, device='cuda', dtype=torch.float)
with torch.backends.cudnn.flags(enabled=False):
    loss_native = torch.nn.functional.ctc_loss(log_probs.double(), targets, input_lengths, target_lengths, reduction='none')
    grad_native = torch.autograd.grad(loss_native, log_probs, grad_out)[0].float()
torch.autograd.gradcheck(lambda x: torch.nn.functional.ctc_loss(x.log_softmax(2), targets, input_lengths, target_lengths, reduction='sum'), log_probs.double())
loss_cudnn = torch.nn.functional.ctc_loss(log_probs, targets.to('cpu', torch.int32),
                                                   input_lengths, target_lengths, reduction='none')
grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
print((grad_cudnn-grad_native).max())