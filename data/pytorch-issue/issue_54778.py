import torch
import torch.utils.benchmark as benchmark

torch.manual_seed(391)
A = torch.randn(100, 100)
A = torch.mm(A, A.t()) + 1e-3 * torch.eye(100)  # make symmetric


### CPU ###

res1 = benchmark.Timer(
    stmt="torch.cholesky(A)",
    setup="import torch",
    globals=dict(A=A),
    num_threads=torch.get_num_threads(),
    label='Cholesky (cpu)',
    sub_label='single',
    description='time',
).blocked_autorange(min_run_time=1)

res2 = benchmark.Timer(
    stmt="torch.cholesky(A)",
    setup="import torch",
    globals=dict(A=A.unsqueeze(0)),
    num_threads=torch.get_num_threads(),
    label='Cholesky (cpu)',
    sub_label='batch',
    description='time',
).blocked_autorange(min_run_time=1)

compare = benchmark.Compare([res1, res2])
compare.print()


### CUDA ###

A = A.cuda()

res1 = benchmark.Timer(
    stmt="torch.cholesky(A)",
    setup="import torch",
    globals=dict(A=A),
    num_threads=torch.get_num_threads(),
    label='Cholesky (cuda)',
    sub_label='single',
    description='time',
).blocked_autorange(min_run_time=1)

res2 = benchmark.Timer(
    stmt="torch.cholesky(A)",
    setup="import torch",
    globals=dict(A=A.unsqueeze(0)),
    num_threads=torch.get_num_threads(),
    label='Cholesky (cuda)',
    sub_label='batch',
    description='time',
).blocked_autorange(min_run_time=1)

compare = benchmark.Compare([res1, res2])
compare.print()