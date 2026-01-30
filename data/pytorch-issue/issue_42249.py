print('before import')
import torch
print('after import')
print('devices: ', torch.cuda.device_count())
x = torch.tensor([1,2,3])
print('tensor creation')
x = x.cuda()
print('moved to cuda')