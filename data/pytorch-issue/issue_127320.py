import torch
from torch._higher_order_ops.while_loop import while_loop
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def test_while_loop_tpu_MNIST_inside_loop(self):

    torch.set_grad_enabled(False)

    n_epochs = 3
    batch_size_train = 8
    batch_size_test = 10
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    class MNIST(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(500, 50)
        self.fc2 = torch.nn.Linear(50, 10)

      def forward(self, iteri, x, y):
        def cond_fn(iteri, x, y):
          return iteri > 0

        def body_fn(iteri, x, y):
          y = F.relu(F.max_pool2d(self.conv1(x), 2))
          y = self.bn1(y) # torch.while_loop's body_fn might be modifying the input!
          y = F.relu(F.max_pool2d(self.conv2(y), 2))
          y = self.bn2(y)
          y = torch.flatten(y, 1)
          y = F.relu(self.fc1(y))
          y = self.fc2(y)

          return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

        return while_loop(cond_fn, body_fn, (iteri, x, y))

      def forward_compare(self, iteri, x, y):
        y = F.relu(F.max_pool2d(self.conv1(x), 2))
        y = self.bn1(y) # torch.while_loop's body_fn might be modifying the input!
        y = F.relu(F.max_pool2d(self.conv2(y), 2))
        y = self.bn2(y)
        y = torch.flatten(y, 1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return iteri - 1, x.clone(), F.log_softmax(y, dim=1)

    mnist = MNIST()
    bs=16
    l_in_0 = torch.randn(bs, 1, 28, 28, dtype=torch.float32)
    l_out = torch.randn(bs, 10, dtype=torch.float32)
    iteri = torch.tensor(3, dtype=torch.int64)
    _, _, res = mnist(iteri, l_in_0, l_out)

    # === expected result for one iteration to be compared since body_fn defined use the same input in each iteration ===
    _, _, expected_res = mnist.forward_compare(iteri, l_in_0, l_out)
    self.assertTrue(torch.all(torch.eq(res, expected_res)))

ep = torch.export.export(mnist, (iteri, l_in_0, l_out))
ep.module().print_readable()