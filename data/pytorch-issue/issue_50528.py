import torch.nn as nn

import torch

batch_size, num_class = 8, 10
logits = torch.randn(batch_size, num_class)
target = torch.randint(num_class, size=[batch_size])
ce_loss = torch.nn.CrossEntropyLoss
class_weight = torch.rand(num_class)
ce_loss(weight=class_weight)(logits, target)
ce_loss(weight=class_weight, reduction='none')(logits, target).mean()

import torch

batch_size, num_class = 1, 10
logits = torch.randn(batch_size, num_class)
target = torch.randint(num_class, size=[batch_size])
ce_loss = torch.nn.CrossEntropyLoss
class_weight = torch.rand(num_class)
ce_loss()(logits, target)
ce_loss(weight=class_weight)(logits, target)