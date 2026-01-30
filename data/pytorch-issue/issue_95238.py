import torch

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], with_stack=True) as prof:
      self._training_step(...)

prof.export_stacks("stacks.txt", "self_cuda_time_total")

...
loss = model(**X, **y).loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
...