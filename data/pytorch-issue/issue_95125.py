import torch.nn as nn

import torch

def check_correctness(a, b, expected):
    for mkldnn_flag in [True, False]:
        with torch.backends.mkldnn.flags(enabled=mkldnn_flag):
            def test():
                tmp_result= torch.linalg.matmul(a, b)
                # tmp_result= torch.Tensor.matmul(a, b)
                return tmp_result
            c = test()
            assert(torch.all(c == expected)), "Incorrect result with\n" \
                                              f"torch.backends.mkldnn.flags(enabled={mkldnn_flag}),\n" \
                                              f"and dtypes: {a.dtype}, {b.dtype}, {c.dtype}\n" \
                                              f"expected: {expected}\n" \
                                              f"got: {c}\n"

val = 1024
a = torch.ones(val, val)
b = torch.ones(val, val)

check_correctness(a, b, expected=val)

a = a.to(torch.bfloat16)
b = b.to(torch.bfloat16)

check_correctness(a, b, expected=val)