import torch as tc
print(tc.__version__) # 0.2.0
t = tc.ones(3)
tc.save(t, './t.pth')

import torch as tc
print(tc.__version__) # 0.4.0a0+67bbf58
t = tc.load('./t.pth') # error