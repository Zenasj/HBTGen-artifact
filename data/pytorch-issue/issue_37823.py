import torch
import torch.nn as nn

bn = nn.BatchNorm1d(4)
bn.track_running_stats = False

# Store initial values
num_batches = bn.num_batches_tracked.clone() 
running_mean = bn.running_mean.clone() 
running_var = bn.running_var.clone()

# Forward random tensor
_ = bn(torch.rand(32, 4))

# Check which stats were updated
print(torch.equal(num_batches, bn.num_batches_tracked))
print(torch.equal(running_mean, bn.running_mean))
print(torch.equal(running_var, bn.running_var))

True
False
False

py
(self.running_mean is None and self.running_var is None) if self.track_running_stats else self.training

(self.running_mean is None and self.running_var is None) if not self.track_running_stats else self.training

self.training if self.track_running_stats else (self.running_mean is None and self.running_var is None)