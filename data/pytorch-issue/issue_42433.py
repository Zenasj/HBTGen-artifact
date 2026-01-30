import timeit
import torch
runtimes = []
threads = [1] + [t for t in range(2, 49, 2)]
for t in threads:
    torch.set_num_threads(t)
    r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
    runtimes.append(r)
    print("{} is done.Time is {}".format(t,r))

1 is done.Time is 15.510971342213452
2 is done.Time is 15.48298014793545
4 is done.Time is 15.502299554878846
6 is done.Time is 15.491661492967978
8 is done.Time is 15.499503097962588
10 is done.Time is 15.52067736000754
...