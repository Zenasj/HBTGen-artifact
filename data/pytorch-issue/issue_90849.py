import torch

# save the current state
e = torch._C._check_sparse_tensor_invariants()
# override the unsafe state (unless there is a flag that forces the check)
torch._C._set_check_sparse_tensor_invariants(False)
# perform computations involving unsafe tensor constructors
...
# restore the previous state
torch._C._set_check_sparse_tensor_invariants(e)