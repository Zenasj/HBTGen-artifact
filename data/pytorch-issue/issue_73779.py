3
import torch
from timeit import default_timer as timer

vals = torch.arange(1, 10000000, 1).cuda() # Create an array of values
times_tadd = []
times_add = []
times_seq = []
times_pre = []
loops = 10
loops_per_type = 100

for _ in range(loops):
    
    # Using torch add
    s_tadd = timer()
    for _ in range(loops_per_type):
        _vals = torch.add(vals, 10 + 10 + 10)
    e_tadd = timer()
    times_tadd.append([s_tadd, e_tadd])

    # Chaining the operations in a single line
    s_add = timer()
    for _ in range(loops_per_type):
        _vals = vals + 10 + 10 + 10
    e_add = timer()
    times_add.append([s_add, e_add])

    # Performing the operations sequentially
    s_seq = timer()
    for _ in range(loops_per_type):
        _vals = vals + 10
        _vals = _vals + 10
        _vals = _vals + 10  
    e_seq = timer()
    times_seq.append([s_seq, e_seq])

    # Adding the numbers separately first
    s_pre = timer()
    for _ in range(loops_per_type):
        precompute = 10 + 10 + 10
        _vals = vals + precompute
    e_pre = timer()
    times_pre.append([s_pre, e_pre])
    
times_tadd = torch.tensor(times_tadd)
times_add = torch.tensor(times_add)
times_seq = torch.tensor(times_seq)
times_pre = torch.tensor(times_pre)

print('tadd', torch.mean(times_tadd[:, 1] - times_tadd[:, 0]))
print('add', torch.mean(times_add[:, 1] - times_add[:, 0]))
print('seq', torch.mean(times_seq[:, 1] - times_seq[:, 0]))
print('pre', torch.mean(times_pre[:, 1] - times_pre[:, 0]))

3
_vals = 10 + 10 + 10 + vals