import torch

print(torch.__version__)
a = torch.Tensor([1, 2, 2, 3])
print("a: ", a)
b = a.to('mps')
print(f"a.max(): {a.max()}, b.max(): {b.max()}")

c = a.to('mps').to(torch.int)
d = c.to('cpu')
print("c: ", c, "d: ", d)
print(f"a.max(): {a.max()}, c.max(): {c.max()}, d.max(): {d.max()}")

print('============')
c = a.to('mps').to(int)
d = c.to('cpu')
print("c: ", c, "d: ", d)
print(f"a.max(): {a.max()}, c.max(): {c.max()}, d.max(): {d.max()}")

print('============clear')
a = torch.Tensor(
    [[1.0000, 0.0000, 377.2475, 237.9320, 640.0000, 420.5970],
     [2.0000, 0.0000, 523.8516, 55.7052, 640.0000, 219.5220],
     [2.0000, 0.0000, 367.9471, 78.8564, 455.8632, 198.4222],
     [3.0000, 0.0000, 46.5807, 58.2641, 466.7800, 439.6348]],
).to('mps')
b = a[:, 0]
c = b.unique(return_counts=True)[1].cpu().max()
d = b.unique(return_counts=True)[1].max()
print(f"b: {b}\nc: {c}\nd: {d}")