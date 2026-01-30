square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
avg = square_avg.sqrt().add_(group['eps'])
if group['momentum'] > 0:
    buf = state['momentum_buffer']
    buf.mul_(group['momentum']).addcdiv_(grad, avg)
    p.data.add_(-group['lr'], buf)
else:
    p.data.addcdiv_(-group['lr'], grad, avg)

square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
avg = square_avg.sqrt().add_(group['eps'])
if group['momentum'] > 0:
    eff_lr = avg.reciprocal().mul(group['lr'])
    buf = state['momentum_buffer']
    buf.mul_(group['momentum']).addcmul_(eff_lr, grad)
    p.data.add_(-1, buf)