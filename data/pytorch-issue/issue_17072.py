import torch.nn as nn

class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.fc1 = nn.Linear(20, 10)

m = TestModule()

# Under the hood, `m._apply()` uses `.data =` to change the data of `m`'s parameters and their gradients
m = m._apply(lambda t: torch.sparse_coo_tensor(torch.zeros([2, 1]), torch.ones([1]), torch.Size([10, 20])))
# After this PR, this fails with "RuntimeError: Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have different types of TensorImpl."

params = torch.tensor([1.5, 1.5]).requires_grad_()
# Change gradient to a sparse tensor
params.grad = torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.]))

grad_saved = params.grad
params.backward(torch.tensor([1.5, 1.5]))
assert id(grad_saved) == id(params.grad)  # This will fail after this PR

import timeit

print(timeit.timeit('''
output, hn = rnn(input, (h0, c0))''',
setup='''
import torch;
rnn = torch.nn.LSTM(input_size=10, hidden_size=20, num_layers=2);
input = torch.randn(5, 3, 10);
h0 = torch.randn(2, 3, 20);
c0 = torch.randn(2, 3, 20);
# warm up
output, hn = rnn(input, (h0, c0));
output, hn = rnn(input, (h0, c0));
output, hn = rnn(input, (h0, c0))''', number=1000))

import timeit

print(timeit.timeit('''
y = x.abs()''',
setup='''
import torch;
x = torch.randn(2, 3).requires_grad_();
# warm up
y = x.abs();
y = x.abs();
y = x.abs();''', number=100000))