import torch

@torch.jit.script
def smt_pred(confs, child, child_sizes, group_offsets,
             threshold, obj,
             height, width, b, a):
    # type: (List[Tensor],List[Tensor],List[Tensor],List[int],Tensor,Tensor,int,int,int,int) -> Tuple[Tensor,Tensor,Tensor]
    size = child_sizes[0][1]
    ch = child[0][1]
    for sg in range(size):
        pass

for sg in range(size.int()):
    pass