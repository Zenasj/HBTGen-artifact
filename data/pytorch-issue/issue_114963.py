from collections import OrderedDict, defaultdict

import torch


def fn() -> None:
    d = dict.fromkeys(['a', 'b'])
    od = OrderedDict.fromkeys(['a', 'b'])
    dd = defaultdict.fromkeys(['a', 'b'])


comp_out = torch._dynamo.optimize(nopython=True)(fn)()