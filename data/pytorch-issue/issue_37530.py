import torch.nn as nn

import torch
from typing import List
class Boxes():
    def __init__(self, tensor):
        self.tensor = tensor
    @classmethod
    def cat(cls, box_lists: List):
        return cls(torch.cat([x.tensor for x in box_lists]))

def f(t: torch.Tensor):
    b = Boxes(t)
    c = Boxes(torch.tensor([3, 4]))
    return Boxes.cat([b, c])

if __name__ == "__main__":
    f_script = torch.jit.script(f)

def try_ann_to_type(ann, loc):
    if ann is None:
        return TensorType.get()
    elif ann is torch.Tensor:
        return TensorType.get()
    elif is_tuple(ann):
        return TupleType([try_ann_to_type(a, loc) for a in ann.__args__])
    elif is_list(ann):
        return ListType(try_ann_to_type(ann.__args__[0], loc))