import torch
from typing import List, Union

def fn(x:Union[List[int],]) -> str:
    return 'foo'

scripted = torch.jit.script(fn)

import torch
from typing import List, Union

# remove the union
def fn(x:List[int]) -> str:
    return 'foo'

scripted = torch.jit.script(fn)