import torch

from torch.utils.tensorboard import SummaryWriter

with SummaryWriter() as w:
    for i in range(20):
        w.add_hparams(hparam_dict={'lr': 0.1 * i, 'batch_siz': 10 * i},
                      metric_dict={'hparam/accuracy': 0.9 ** i, 'hparam/loss': 0.01 * i})
    
writer.close()