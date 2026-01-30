from itertools import zip_longest
for i, (exp_avg, grad, exp_avg_sq) in enumerate(zip_longest(device_exp_avgs, device_grads, device_exp_avg_sqs)):

    if exp_avg is not None and exp_avg.dim() == 0:
        device_exp_avgs[i] = exp_avg.unsqueeze(0)

    if grad is not None and grad.dim() == 0:
        device_grads[i] = grad.unsqueeze(0)

    if exp_avg_sq is not None and exp_avg_sq.dim() == 0:
        device_exp_avg_sqs[i] = exp_avg_sq.unsqueeze(0)