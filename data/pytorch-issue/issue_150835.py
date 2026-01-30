import torch.nn as nn

import torch
import torch.nn.functional as F
device = "cuda" # "cpu" is fine
num_classes = 4  
log_probs = torch.rand(0, 0, num_classes, device=device)  
targets = torch.tensor([], device=device, dtype=torch.long)
input_lengths = torch.tensor([], device=device, dtype=torch.long) 
target_lengths = torch.tensor([], device=device, dtype=torch.long) 
result = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')