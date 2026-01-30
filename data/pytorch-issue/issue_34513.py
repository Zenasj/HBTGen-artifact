import torch
import torch.nn as nn

m = torch.nn.RNN(10,20,2)
m = torch.jit.script(m)
m = m.cuda() #moving after scripting, only generic _apply is called, params are not flattened
input = torch.randn(5,3,10, device="cuda")
out = m(input) #Warning: parameters are not flattened ...

m = torch.nn.RNN(10,20,2).cuda()
m = torch.jit.script(m)
m = torch.nn.DataParallel(m, [0,1]) # many things went wrong here. overriden _replicate_for_data_parallel was not called, so _flat_weights and _flat_weights_names lists were not copied. Overriden __setattr__ was not called, so _flat_weights is pointing to some wrong tensors with wrong history
input = torch.randn(6,4,10)
out = m(input) #RuntimeError: parameters are on the wrong device ...