import torch as t
from matplotlib import pyplot as plt

complex_param = t.tensor([1],dtype=t.complex64, requires_grad=True)

# We will optimize on the mean squared error from a 1j
target = 1j

def calc_loss(x):
    return t.abs(x - target)**2

optimizer = t.optim.Adam([complex_param], lr=0.001)

n = 10000
values = t.zeros(n, dtype=t.complex64)
for i in range(n):
    optimizer.zero_grad()
    loss = calc_loss(complex_param)
    loss.backward()
    optimizer.step()
    values[i] = complex_param.detach()


# Plot the results
plt.plot(values.real, label='Real Part')
plt.plot(values.imag, label='Imaginary Part')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Complex Parameter')
plt.title('Optimization Progress with as-implemented Adam')
plt.show()

exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)