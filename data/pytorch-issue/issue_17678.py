import torch
import numpy as np


# Moving to double should reduce the difference to something very small ~1e-10
used_type = torch.float
# used_type = torch.double

# Changing the number of threads will potentially change the order of execution
# And thus change all the results
num_threads = 12

torch.manual_seed(1)
torch.set_num_threads(num_threads)
big_size = 1000000
a = torch.randn((big_size, 20), dtype=used_type)
print("Running with type {} and {} threads".format(used_type, num_threads))

sum_a = a.sum().item()
sum_a_np = np.sum(a.numpy())
print("original diff: ", sum_a - sum_a_np)

print("diffs: torch vs torch \t| torch vs np   \t| np vs np")
for _ in range(20):
    idx = torch.randperm(big_size)
    shuffled_a = a.index_select(0, idx)
    new_sum_a = shuffled_a.sum().item()
    new_sum_a_np = np.sum(shuffled_a.numpy())
    print("diffs: {} \t| {} \t| {}".format(sum_a - new_sum_a, new_sum_a - new_sum_a_np, sum_a_np - new_sum_a_np))