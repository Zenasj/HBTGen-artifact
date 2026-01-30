import torch
import numpy as np

seq = pd.Series([1.0, 2.0, 3.0])
torch.Tensor(seq)  # succeeds, since seq[0] is defined
torch.Tensor(seq[1:])  # segfault, since seq[0] generates a KeyError

df = pd.DataFrame(np.ones((2, 3)), columns=['a', 'b', 'c'])
torch.Tensor(df)  # segfault, since df[0] tries to access a column named 0