import torch
from typing import List

@torch.jit.script
def merge_levels(levels, unmerged_results: List[torch.Tensor]):
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros((levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3)),
                      dtype=dtype, device=device)
    for l in range(len(unmerged_results)):
        mask = (levels == l).view(-1, 1, 1, 1).expand(levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3))
        res.masked_scatter_(mask, unmerged_results[l])
    return res

@torch.jit.script
def merge_levels_not(levels, unmerged_results: List[torch.Tensor]):
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros((levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3)),
                      dtype=dtype, device=device)
    print ("inside merge level not######", levels.shape, res.shape)
    for l in range(len(unmerged_results)):
        mask = (levels == l).view(-1, 1, 1, 1).expand(levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3))
        print ("lnot#######", l, res.shape, mask.shape, unmerged_results[l].shape)
    return res

def myscript1(levels):
    unmerged_results = []
    for level in range(5):
        idx_in_level = torch.nonzero(levels == level).view(-1, 1, 1, 1).expand(-1, 3, 14, 14)
        unmerged_results.append(idx_in_level)
    
    res = merge_levels_not(levels, unmerged_results)
    return res

def myscript2(levels):
    unmerged_results = []
    for level in range(5):
        idx_in_level = torch.nonzero(levels == level).view(-1, 1, 1, 1).expand(-1, 3, 14, 14)
        unmerged_results.append(idx_in_level)

    res = merge_levels(levels, unmerged_results)
    return res

levels = torch.randint(0,5, (50,))

scr = torch.jit.trace(myscript1, (levels,))
scr2 = torch.jit.trace(myscript2, (levels,))
print (scr.graph_for(levels))
print (scr2.graph_for(levels))
levels2 = torch.randint(0,5, (55,))

scr(levels2)
scr2(levels2)

@torch.jit.script                                                  
def merge_levels(levels, unmerged_results):                                            
     first_result = unmerged_results                                                    
     dtype, device = first_result.dtype, first_result.device                            
     res = torch.zeros((levels.size(0), first_result.size(1),                           
                        first_result.size(2), first_result.size(3)),                    
                       dtype=dtype, device=device)                                      
     mask = (levels == 0).view(-1, 1, 1, 1).expand(levels.size(0), first_result.size(1),
                        first_result.size(2), first_result.size(3)) 
     res.masked_scatter_(mask, unmerged_results)                    
     return res