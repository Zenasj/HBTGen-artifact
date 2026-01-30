import torch
def test(a,b):
    print()
    print(a/b)
    print(torch.tensor(a)/b)
    print(a/torch.tensor(b))        # Potential bug.
    print(a/float(torch.tensor(b))) # Potential much more precision loss than merely float() casting.
    print(a/float(b))
test(6.10840013583831e-41, 2.7734e-39)  # Bug to inf. (original found case)
test(0.0, 2.8026e-45)                   # Bug to nan. (original found case)
test(1.0e-39, 1.0e-39)                  # Bug to inf and more precision loss.
test(1.0e-38, 1.0e-39)                  # Bug to inf and more precision loss.
test(1.0e-39, 1.0e-38)                  # More precision loss.
test(1.0e-38, 1.0e-38)                  # More precision loss.
test(0.0, 1.0e-39)                      # Bug to nan.
test(0.0, 1.0e-38)

0.022024951813075323
tensor(0.0220)
tensor(inf)
0.022024955991519653
0.022024951813075323

0.0
tensor(0.)
tensor(nan)
0.0
0.0

1.0
tensor(1.)
tensor(inf)
0.9999997846947131
1.0

10.0
tensor(10.0000)
tensor(inf)
9.99999784694713
10.0

0.09999999999999999
tensor(0.1000)
tensor(0.1000)
0.10000000649543637
0.09999999999999999

1.0
tensor(1.)
tensor(1.)
1.0000000649543639
1.0

0.0
tensor(0.)
tensor(nan)
0.0
0.0

0.0
tensor(0.)
tensor(0.)
0.0
0.0