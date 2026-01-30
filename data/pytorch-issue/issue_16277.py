import torch.nn as nn

python
self.conv1.stddev = 0.01

python
if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
   import scipy.stats as stats
   stddev = m.stddev if hasattr(m, 'stddev') else 0.1

python
if hasattr(m, 'stddev'):
    print(m)

Linear(in_features=768, out_features=1000, bias=True)