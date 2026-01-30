import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

fig = plt.figure()
plt.plot([1,2])
fig.savefig('test.pdf')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"