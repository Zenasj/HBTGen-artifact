import torch

from torch.testing._comparison import assert_equal, TensorLikePair, ObjectPair

assert_equal("a", "a", pair_types=(TensorLikePair, ObjectPair))