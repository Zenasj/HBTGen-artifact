import torch

def _pad_mm_init():
    from .joint_graph import patterns

    #if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
    #    device = "cuda"
    #else:
    device = "cpu"