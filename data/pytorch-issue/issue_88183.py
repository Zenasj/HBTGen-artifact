import torch.nn as nn

def main():
    import torch
    import torch.nn.functional as F
    
    # construct tensor on cpu
    t = torch.ones([1, 1, 1, 2])
    t = t.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=[2, 2], mode="bilinear", align_corners=False)

    #construct on mps
    t_mps = torch.ones([1, 1, 1, 2], device="mps")
    t_mps = t_mps.permute(0, 3, 1, 2).contiguous() # even applying contiguous() doesn't fix the problem
    t_mps = F.interpolate(t_mps, size=[2, 2], mode="bilinear", align_corners=False)

    # construct directly on mps with shape after permute
    t_mps_no_permute = torch.ones([1, 2, 1, 1], device="mps")
    t_mps_no_permute = F.interpolate(t_mps_no_permute, size=[2, 2], mode="bilinear", align_corners=False)

    print("cpu result: ", t)
    print("mps result: ", t_mps.cpu())
    print("mps no permute result: ", t_mps_no_permute.cpu())

    return

t_mps = t_mps.permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)