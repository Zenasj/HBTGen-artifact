import torch

@torch.no_grad()
def _dequeue_and_enqueue(self, keys):
    # gather keys before updating queue
    keys = concat_all_gather(keys)

    batch_size = keys.shape[0]

    ptr = int(self.queue_ptr)
    assert self.K % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    self.queue[:, ptr:ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % self.K  # move pointer

    self.queue_ptr[0] = ptr