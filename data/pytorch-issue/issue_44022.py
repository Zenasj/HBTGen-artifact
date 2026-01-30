import torch.nn.functional as F

self.layer_weights = F.softmax(self.layer_weights,dim=0)