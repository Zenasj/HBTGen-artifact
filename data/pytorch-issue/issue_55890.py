import torch
import numpy as np

def make_msg(a, b, info):
    return (
        f"Argh, we found {info.total_mismatches} mismatches! "
        f"That is {info.mismatch_ratio:.1%}!"
    )

torch.testing.assert_equal(torch.tensor(1), torch.tensor(2), msg=make_msg)

def make_msg(input, torch_output, numpy_output, info):
    return (
        f"For input {input} torch.binary_op() and np.binary_op() do not match: "
        f"{torch_output} != {numpy_output}"
    )

torch.testing.assert_equal(
    torch.binary_op(input),
    numpy.binary_op(input),
    msg=lambda a, b, info: make_msg(input, a, b, info),
)