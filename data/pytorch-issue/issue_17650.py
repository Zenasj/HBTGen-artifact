import torch

py
@torch.jit.script
def MyScriptFun1(input1:Tuple[List[int]]) -> Tuple[List[int]]:
    return input1