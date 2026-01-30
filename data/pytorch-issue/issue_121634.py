import torch.nn as nn

import torch

def ctc_cal():
    arg1 = torch.rand([50, 3, 15], dtype=torch.float64)
    log_probs = arg1.clone()
    arg2 = torch.randint(0, 2, [3, 30], dtype=torch.int64)
    targets = arg2.clone()
    input_lengths = [50,50,50]
    target_lengths = [30,25,20]
    blank = 14
    reduction = "mean"
    zero_infinity = False
    res = torch.nn.functional.ctc_loss(log_probs=log_probs,
                                       targets=targets,
                                       input_lengths=input_lengths,
                                       target_lengths=target_lengths,
                                       blank=blank,
                                       reduction=reduction,
                                       zero_infinity=zero_infinity,)

    return res

# without torch.jit.script
result = ctc_cal()
print(result)

# # with torch.jit.script
# scripted_ctc_cal = torch.jit.script(ctc_cal)
# result = scripted_ctc_cal()
# print(result)