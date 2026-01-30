import torch.nn as nn

import torch
print(torch.__version__)  #should return 1.12.1
print(torch.cuda.is_available())  #should return  True
print(torch.backends.cudnn.enabled) #should return True