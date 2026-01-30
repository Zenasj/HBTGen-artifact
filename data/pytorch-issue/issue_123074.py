exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

fused_kernel(kwargs1)

# convert to float for lowp
for arg in kwargs2:
    arg.to(float)
non_fused_kernel(kwargs2)
# convert to low precision
for arg in kwargs2:
    arg.to(bfloat16)

assertEqual(kwargs1, kwargs2)