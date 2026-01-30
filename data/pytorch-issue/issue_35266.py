import torch

# On all ranks
with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(3, 3, requires_grad=True)
            loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2)).sum()
            with torch.autograd.profiler.profile() as p:
                dist_autograd.backward(context_id, [loss])