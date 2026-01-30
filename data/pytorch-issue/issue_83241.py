import torch.nn as nn

import torch
results={}
arg_1 = -7.0
arg_2 = 2
arg_class = torch.nn.TripletMarginLoss(margin=arg_1,p=arg_2)
arg_3_0 = torch.rand([100, 128], dtype=torch.float32)
arg_3_1 = torch.rand([100, 128], dtype=torch.float32)
arg_3_2 = torch.rand([100, 128], dtype=torch.float32)
arg_3 = [arg_3_0,arg_3_1,arg_3_2]
results['res'] = arg_class(*arg_3)