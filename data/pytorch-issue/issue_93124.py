import torch
torch.linalg.eigvals(torch.tensor([[1.1300,    torch.nan], [torch.nan,    torch.nan]]))

"""
Intel MKL ERROR: Parameter 3 was incorrect on entry to SGEBAL.

Intel MKL ERROR: Parameter 2 was incorrect on entry to SGEHD2.

Intel MKL ERROR: Parameter 4 was incorrect on entry to SHSEQR.
python: malloc.c:2379: sysmalloc: Assertion `(old_top == initial_top (av) && old_size == 0) || ((unsigned long) (old_size) >= MINSIZE && prev_inuse (old_top) && ((unsigned long) old_end & (pagesize - 1)) == 0)' failed.
[1]    1603318 abort (core dumped)  ipython
"""