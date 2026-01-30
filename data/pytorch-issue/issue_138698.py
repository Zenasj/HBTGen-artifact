import torch

t = torch.rand((2, 8, 2, 2))
inList = (t, t, t)
cinputs = {'self':inList, 'exponent':2.7}
fnc = torch._foreach_pow
fnc = torch.compile(fnc, backend="inductor")  # "eager" fails too

res1 = fnc(inList, 2.7)
res2 = fnc(self=inList, exponent=2.7)  # fails since PT2.5
res3 = fnc(**cinputs)                            # fails since PT2.5