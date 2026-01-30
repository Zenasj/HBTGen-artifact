p -= weight_decay * p

p -= lr * weight_decay * p

if weight_decay != 0:
    p.data.add_(-weight_decay, p.data) # p.data = p.data - weight_decay * p.data

p.data.add_(-group['lr'], d_p) # p.data = p.data - lr * d_p = p.data -lr * d_p - weight_decay * p.data