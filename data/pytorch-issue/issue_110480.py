import torch.nn as nn

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


# Linear learning rate decay function
def linear_lr_decay(epoch, max_epochs, lr_start, lr_end):
    slope = (lr_end - lr_start) / max_epochs
    return lr_start + slope * epoch

# Number of epochs, initial/final learning rates
num_epochs = 100
lr_start = 4e-4
lr_end = 1e-5

# Dummy optimizer (replace with your own optimizer)
model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=lr_start)

# Create a LambdaLR scheduler with linear decay
scheduler = LambdaLR(optimizer, lambda epoch: linear_lr_decay(epoch, num_epochs, lr_start, lr_end))
# scheduler = LinearLR(optimizer, start_factor=0.8, total_iters=num_epochs)


# Simulate training and log the learning rates
learning_rates = []
for epoch in range(num_epochs):
    # Step the scheduler to update the learning rate
    learning_rates.append(scheduler.get_last_lr()[0])
    optimizer.step()
    scheduler.step()

# Plotting the learning rate
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), learning_rates, label='Linear Learning Rate Schedule', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Linear Learning Rate Schedule')
plt.legend()
plt.grid(True)
plt.show()

4.00e-04
3.96e-04
3.92e-04
3.88e-04
3.84e-04
3.80e-04
3.76e-04
3.72e-04
3.68e-04
3.65e-04
3.61e-04
3.57e-04
3.53e-04
3.49e-04
3.45e-04
3.41e-04
3.37e-04
3.33e-04
3.29e-04
3.25e-04
3.21e-04
3.17e-04
3.13e-04
3.09e-04
3.05e-04
3.02e-04
2.98e-04
2.94e-04
2.90e-04
2.86e-04
2.82e-04
2.78e-04
2.74e-04
2.70e-04
2.66e-04
2.62e-04
2.58e-04
2.54e-04
2.50e-04
2.46e-04
2.42e-04
2.38e-04
2.35e-04
2.31e-04
2.27e-04
2.23e-04
2.19e-04
2.15e-04
2.11e-04
2.07e-04
2.03e-04
1.99e-04
1.95e-04
1.91e-04
1.87e-04
1.83e-04
1.79e-04
1.75e-04
1.72e-04
1.68e-04
1.64e-04
1.60e-04
1.56e-04
1.52e-04
1.48e-04
1.44e-04
1.40e-04
1.36e-04
1.32e-04
1.28e-04
1.24e-04
1.20e-04
1.16e-04
1.12e-04
1.08e-04
1.05e-04
1.01e-04
9.67e-05
9.27e-05
8.88e-05
8.48e-05
8.09e-05
7.70e-05
7.30e-05
6.91e-05
6.52e-05
6.12e-05
5.73e-05
5.33e-05
4.94e-05
4.55e-05
4.15e-05
3.76e-05
3.36e-05
2.97e-05
2.58e-05
2.18e-05
1.79e-05
1.39e-05
1.00e-05