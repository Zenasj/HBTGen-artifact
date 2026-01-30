import torch

from torch.utils.data.datapipes.map import SequenceWrapper
from torch.utils.data.datapipes.iter import ShardingFilter, Batcher
from torch.utils.data.graph_settings import apply_sharding

dp = SequenceWrapper(range(100))  # len(dp) is 100
dp = ShardingFilter(dp)  # len(dp) is 100
dp = Batcher(dp, 8)    # len(dp) is 13
len(dp)        # <- the issue—this caches the length of `Batcher`.
apply_sharding(dp, num_of_instances=2, instance_id=0)  # len(dp) should be 7 now (50 samples per shard, makes 6 full batches and one of 2 samples)
print(len(dp))  # Prints 13 rather than 7.

def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                length = len(self.datapipe) // self.batch_size
            else:
                length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return length
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))