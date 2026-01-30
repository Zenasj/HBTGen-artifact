import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, Adadelta


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, obs, act):
        return self.q(obs)

def sac(device, lr=3e-4, weight_decay=0.0):
    q_function = MLPQFunction(20, 256)
    q_function.to(device)
    q_optimizer = Adam(q_function.parameters(), lr=lr, weight_decay=weight_decay)
    start = time.perf_counter()
    last = start
    for j in range(1000000):
        q_optimizer.zero_grad()
        o, a = torch.rand(256, 20, device=device), torch.rand(256, 20, device=device)
        q = q_function(o,a)
        backup = torch.rand(256, device=device)
        loss_q = ((q - backup)**2).mean()
        loss_q.backward()
        q_optimizer.step()
        if j % 500 == 0:
            curr = time.perf_counter()
            print(f'Iteration: {j}, Time: {curr - start:.2f}s')
            print(f'Time since last iteration: {curr - last:.2f}s')
            last = curr


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        torch.set_num_threads(2)

    try:
        sac(device, weight_decay=args.weight_decay)
    except KeyboardInterrupt:
        pass