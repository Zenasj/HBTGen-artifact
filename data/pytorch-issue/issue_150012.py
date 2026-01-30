sample_165 = torch.tensor([[18.3906, 18.3906, 17.5938, 17.9844, 15.1172, 18.3594, 18.3438, 15.7812, 17.8438, 17.6719]], device='cuda:0', dtype=torch.float16)

import torch

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc, pred

# Single sample processing
sample_165 = tensor([[18.3906, 18.3906, 17.5938, 17.9844, 15.1172, 18.3594, 18.3438, 15.7812, 17.8438, 17.6719]], device='cuda:0', dtype=torch.float16)
_, single_pred = cls_acc(sample_165, torch.tensor([1], device='cuda:0'))
print(f"Single sample prediction index: {single_pred.item()}")  # Returns 0

# Batch processing
tot_logits = torch.load('tot_logits.pt')  # A tensor containing sample_165 at index 165
tot_targets = torch.load('tot_targets.pt')
_, pred = cls_acc(tot_logits, tot_targets)
print(f"Batch processing prediction index for sample 165: {pred[0, 165].item()}")  # Returns 1