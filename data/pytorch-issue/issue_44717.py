import torch

from torch.utils._benchmark import Timer
counts = Timer(
    "x.backward()",
    setup="x = torch.ones((1,)) + torch.ones((1,), requires_grad=True)"
).collect_callgrind()

for c, fn in counts[:20]:
    print(f"{c:>12}  {fn}")

print(f"Head instructions: {sum(c for c, _ in counts)}")
print(f"1.6 instructions:  {sum(c for c, _ in counts_1_6)}")
count_dict = {fn: c for c, fn in counts}
for c, fn in counts_1_6:
    _ = count_dict.setdefault(fn, 0)
    count_dict[fn] -= c
count_diffs = sorted([(c, fn) for fn, c in count_dict.items()], reverse=True)
for c, fn in count_diffs[:15] + [["", "..."]] + count_diffs[-15:]:
    print(f"{c:>8}  {fn}")