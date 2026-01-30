import torch

high_bits_for_seed = 16000000000000000000           # to use "good quality" seed
_ = torch.manual_seed (high_bits_for_seed + 2024)

prob = torch.ones (26)
dups_mult = 0
perm_counts_mult = {}
for _ in range (1_000_000):
    p = tuple (torch.multinomial (prob, prob.numel(), replacement=False).tolist())
    if  p in perm_counts_mult:
        dups_mult += 1
        perm_counts_mult[p] += 1
    else:
        perm_counts_mult[p] = 1

print ('duplicate multinomial perms: ', dups_mult)
print ('multiple multinomial perms:  ', (torch.tensor (list (perm_counts_mult.values())) > 1).sum().item())
print ('max of perm_counts_mult:     ', torch.tensor (list (perm_counts_mult.values())).max().item())
print ('len (perm_counts_mult):      ', len (perm_counts_mult))