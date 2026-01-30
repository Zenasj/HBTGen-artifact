import itertools
import math

import torch


input = 90
expected = math.pi / 2


for type, dynamic in itertools.product([float, int], [False, True]):
    torch._dynamo.reset()
    cfn = torch.compile(fullgraph=True, dynamic=dynamic)(math.radians)

    try:
        torch.testing.assert_close(cfn(type(input)), expected)
        result = "PASS"
    except:
        result = "FAIL"
    print(f"{result} {type=}, {dynamic=}")

assert isinstance(math.radians(90), float)