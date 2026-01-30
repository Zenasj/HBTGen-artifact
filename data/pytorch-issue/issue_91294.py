import torch

def metrics(Y, Ypred):
    pw_cmp = (Y == Ypred).float()
    # batch-wise pair-wise overlap rate
    batch_overlap_rate = pw_cmp.mean(dim=0)
    
    # overlap_rate and absolute accuracy
    overlap_rate = batch_overlap_rate.mean().item()
    abs_correct = (batch_overlap_rate == 1.0)
    abs_accu = abs_correct.float().mean().item()

print(batch_overlap_rate[0])
print(batch_overlap_rate2[0])
print(batch_overlap_rate[0] == torch.tensor(1.))
print(batch_overlap_rate2[0] == torch.tensor(1.))

# tensor(1.0000, device='cuda:0')
# tensor(1., device='cuda:0') 
# tensor(False, device='cuda:0')
# tensor(True, device='cuda:0')

print(batch_overlap_rate[0] - 1)
print(batch_overlap_rate2[0] - 1)

# tensor(-5.9605e-08, device='cuda:0')
# tensor(0., device='cuda:0')