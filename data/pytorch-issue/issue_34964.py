with torch.no_grad():
    net.weights.add_(-lr, net.weights.grad)

import torch
import torch.nn as nn
import gc

def print_stats(i):
    print('\nIteration %d' % i)
    print('Resources:')
    for obj in gc.get_objects():  # as in https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/2
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print('\t', type(obj), obj.size())
        except:
            pass

    print('\t{0: <15}\t{1: >6} MB'.format('Allocated', torch.cuda.memory_allocated(device) // 1024 ** 2))
    print('\t{0: <15}\t{1: >6} MB'.format('Max allocated', torch.cuda.max_memory_allocated(device) // 1024 ** 2))
    print('\t{0: <15}\t{1: >6} MB'.format('Cached', torch.cuda.memory_cached(device) // 1024 ** 2))
    print('\t{0: <15}\t{1: >6} MB'.format('Max cached', torch.cuda.max_memory_cached(device) // 1024 ** 2))

def make_sparse_weights(input_length):
    values = torch.ones(input_length, dtype=torch.float)
    indices = torch.tensor([[0, i] for i in range(input_length)], dtype=torch.long).t()
    return torch.sparse.FloatTensor(indices, values, size=(1, input_length))


class SparseNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.weights = nn.Parameter(make_sparse_weights(input_length),
                                    requires_grad=True)

    def forward(self, x):
        return torch.norm(torch.sparse.mm(self.weights, x)) ** 2


class DenseNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.weights = nn.Parameter(make_sparse_weights(input_length).to_dense(), 
                                    requires_grad=True)

    def forward(self, x):
        return torch.norm(torch.mm(self.weights, x)) ** 2

def run_net(net, x, lr, device, n_iterations, 
            update_weight=True, empty_cache=False, collect_garbage=False):
    for i in range(n_iterations):
        # free the gradients and print resources

        if net.weights.grad is not None:
            net.weights.grad.detach_()
            net.weights.grad.zero_()

        if empty_cache:
            torch.cuda.empty_cache()
        if collect_garbage:
            gc.collect()

        if i % 1000 == 0:
            print_stats(i)

        # compute
        loss = net(x)
        loss.backward()

        # update
        if update_weight and net.weights.grad is not None:
            # the SGD implementation with p.data.add_() 
            # throws an error and suggests this solution:

            with torch.no_grad():
                net.weights.add_(-lr, net.weights.grad)

        if i % 1000 == 0:
            print('Loss: %f' % loss.item())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

input_length = 1000
n_iterations = 4001
lr = 1e-6
x = torch.ones(input_length, 1, device=device)

run_net(DenseNetwork(input_length).to(device), x, lr, device, n_iterations, 
        update_weight=True, empty_cache=False, collect_garbage=False)

run_net(SparseNetwork(input_length).to(device), x, lr, device, n_iterations, 
        update_weight=True, empty_cache=False, collect_garbage=False)

run_net(SparseNetwork(input_length).to(device), x, lr, device, n_iterations, 
        update_weight=True, empty_cache=True, collect_garbage=False)

run_net(SparseNetwork(input_length).to(device), x, lr, device, n_iterations, 
        update_weight=False, empty_cache=False, collect_garbage=False)