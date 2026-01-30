import torch as th

rand_cpu_5d = th.randn((2, 1, 32, 32, 32), device="cpu")
print(th.allclose(rand_cpu_5d[0], rand_cpu_5d[1])) # False, as desired (tested multiple times)

rand_mps_5d = th.randn((2, 1, 32, 32, 32), device="mps")
print(th.allclose(rand_mps_5d[0], rand_mps_5d[1])) # True, not desired (tested multiple times)

rand_cpu_4d = th.randn((2, 32, 32, 32), device="cpu")
print(th.allclose(rand_cpu_4d[0], rand_cpu_4d[1])) # False, as desired (tested multiple times)

rand_mps_4d = th.randn((2, 32, 32, 32), device="mps")
print(th.allclose(rand_mps_4d[0], rand_mps_4d[1])) # False, as desired (tested multiple times)