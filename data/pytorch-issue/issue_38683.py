import torch

torch.deterministic = True

torch.deterministic = True

torch.set_deterministic(true_or_false)
...
restore_deterministic = torch.is_deterministic()
torch.set_deterministic(False)
torch.some_nondeterministic_op()
torch.set_deterministic(restore_deterministic)
...

torch.set_deterministic(true_or_false)
...
with torch.flags(deterministic=False):
    torch.some_nondeterministic_op()
...