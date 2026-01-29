# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 32 * 32, 10)  # Assuming input size is 32x32

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 32, 32  # Batch size, channels, height, width
    return torch.rand(B, C, H, W, dtype=torch.float32)

# Custom RMSprop implementation based on TensorFlow's version
class RMSpropTF(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0.9):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super(RMSpropTF, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    eff_lr = avg.reciprocal().mul(group['lr'])
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcmul_(eff_lr, grad)
                    p.data.add_(-1, buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss

# Example usage
# model = my_model_function()
# optimizer = RMSpropTF(model.parameters(), lr=0.001, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0.9)
# input_tensor = GetInput()
# output = model(input_tensor)
# loss = some_loss_function(output, target)
# loss.backward()
# optimizer.step()

