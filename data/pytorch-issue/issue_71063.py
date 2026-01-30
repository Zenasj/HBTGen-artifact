import torch

a, b = torch.randn([5]), torch.randn([5])
batch = [a, b]
elem = batch[0]

# In case torch.utils.data.get_worker_info() is not None
numel = sum(x.numel() for x in batch)
storage = elem.storage()._new_shared(numel)
out = elem.new(storage)

torch.stack(batch, 0, out=out)

import torch

a, b = torch.randn([5]), torch.randn([5])
batch = [a, b]
elem = batch[0]

# In case torch.utils.data.get_worker_info() is not None
numel = sum(x.numel() for x in batch)
storage = elem.storage()._new_shared(numel)
out = elem.new(storage).view(-1, *list(elem.size()))

torch.stack(batch, 0, out=out)