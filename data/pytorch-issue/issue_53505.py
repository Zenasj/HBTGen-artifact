import torch
import torch.nn as nn

asr_loss = ctc_loss(asr_out.log_softmax(2).contiguous(), targets.contiguous(), asr_out_sizes.contiguous(), target_sizes.contiguous())

asr_out.sum()

N = 1
S = 256
C = 10
T = 500
target = torch.randint(low=1, high=C, size=(S,), dtype=torch.int)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int)
target_lengths = torch.tensor(S, dtype=torch.int)
inp = torch.randn(T, N, C, device='cuda').log_softmax(2).requires_grad_()
print(torch.backends.cudnn.version())
torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')