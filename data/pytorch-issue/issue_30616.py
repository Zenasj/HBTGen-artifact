import torch
import torch.nn as nn

if len(self.gpu_ids) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in self.gpu_ids])
    device = torch.device(('cuda:{}').format(self.gpu_ids[0]) if self.gpu_ids else 'cpu')

if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

net.to(device)
image = input['image'].to(device)
label = input['label'].to(device)
pred = net(image)

if len(gpu_ids) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpu_ids])