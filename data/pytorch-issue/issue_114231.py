import itertools

import torch

input = [11, 17]


def f(input, dtype):
    return torch.tensor(input, dtype=dtype)


for backend, dynamic, dtype, input_type in itertools.product(
    ["eager", "inductor"],
    [False, True],
    [torch.int, torch.float],
    [list, tuple],
):
    torch._dynamo.reset()
    cfn = torch.compile(fullgraph=True, dynamic=dynamic, backend=backend)(f)

    output = cfn(input_type(input), dtype)

    try:
        assert output.tolist() == input
        result = "PASS"
    except:
        result = "FAIL"

    print(f"{result} {backend=}, {dynamic=}, {dtype=}, {input_type=}: {output}")