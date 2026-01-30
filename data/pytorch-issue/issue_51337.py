import torch

if use_gpu is None:
    self.activities = set([ProfilerActivity.CPU])
    if torch.cuda.is_available():
        self.activities.add(ProfilerActivity.CUDA)