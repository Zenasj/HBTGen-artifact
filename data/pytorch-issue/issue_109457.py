import torch
import torch.nn as nn
vocab_size = 50304
n_embd = 768
ctxlen = 1024
lm_head = nn.Linear(n_embd, vocab_size, bias=False)

def eval_diff(batch_size):
	torch.manual_seed(1234)
	x = torch.randn(batch_size, ctxlen, n_embd)
	y_cpu = lm_head.to('cpu')(x.to('cpu'))
	y_mps = lm_head.to('mps')(x.to('mps'))
	y_diff = (y_cpu - y_mps.to('cpu'))
	print(f'{batch_size=} -> mean: {y_diff.mean().item():.2f}, std: {y_diff.std().item():.2f}, mean_error_per_cell: {y_diff.abs().mean():.5f}')

eval_diff(batch_size=7) # batch_size=7 -> mean: 0.00, std: 0.00, accumulative_error: 0.00000
eval_diff(batch_size=8) # batch_size=8 -> mean: 0.00, std: 0.00, accumulative_error: 0.00000
# After this point, things will explode: note the cpu-mps difference (their stddev) becomes non-negligible
eval_diff(batch_size=9) # batch_size=9 -> mean: -0.00, std: 0.53, accumulative_error: 0.38577
eval_diff(batch_size=20) # batch_size=20 -> mean: 0.00, std: 0.53, accumulative_error: 0.38575
eval_diff(batch_size=40) # batch_size=40 -> mean: 0.00, std: 0.55, accumulative_error: 0.42468

import torch
import torch.nn as nn
C = 1
T = 1024
for outdim in (20, 50304):
    for B in (1, 10, 20):
        l = nn.Linear(C, outdim, bias=False)
        w = torch.randn(outdim, C)
        x = torch.randn(B, T, C)
        l.weight.data = w
        l_out = {'cpu': l(x), 'mps': l.to('mps')(x.to('mps')).to('cpu')}
        matmul_out = {'cpu': x@w.T, 'mps': (x.to('mps')@w.T.to('mps')).to('cpu')}
        for l_dev in ('cpu', 'mps'):
            for matmul_dev in ('cpu', 'mps'):
                assert l_out[l_dev].allclose(matmul_out[matmul_dev]),\
                    f'nn.Linear in {l_dev} disagrees with manual matrix multiplication in {matmul_dev}: {B=}, {outdim=}, stddev={(l_out[l_dev]-matmul_out[matmul_dev]).std()}'

# AssertionError: nn.Linear in mps disagrees with manual matrix multiplication in cpu: B=10, outdim=50304, stddev=1.4391095638275146

import torch
import torch.nn as nn
vocab_size = 50304
n_embd = 768
ctxlen = 1024
lm_head = nn.Linear(n_embd, vocab_size, bias=False)

def eval_diff(batch_size):
	torch.manual_seed(1234)
	x = torch.randn(batch_size, ctxlen, n_embd)
	y_cpu = lm_head.to('cpu')(x.to('cpu'))
	y_mps = lm_head.to('mps')(x.to('mps'))
	y_diff = (y_cpu - y_mps.to('cpu'))
	print(f'{batch_size=} -> mean: {y_diff.mean().item():.2f}, std: {y_diff.std().item():.2f}, mean_error_per_cell: {y_diff.abs().mean():.5f}')

eval_diff(batch_size=7) # batch_size=7 -> mean: 0.00, std: 0.00, accumulative_error: 0.00000
eval_diff(batch_size=8) # batch_size=8 -> mean: 0.00, std: 0.00, accumulative_error: 0.00000
# After this point, things will explode: note the cpu-mps difference (their stddev) becomes non-negligible
eval_diff(batch_size=9) # batch_size=9 -> mean: -0.00, std: 0.53, accumulative_error: 0.38577
eval_diff(batch_size=20) # batch_size=20 -> mean: 0.00, std: 0.53, accumulative_error: 0.38575
eval_diff(batch_size=40) # batch_size=40 -> mean: 0.00, std: 0.55, accumulative_error: 0.42468