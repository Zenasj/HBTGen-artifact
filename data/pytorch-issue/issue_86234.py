import torch
from torch.utils.benchmark import Timer, Compare

results = []

def test(_sizes):
    x = torch.randn(*_sizes, dtype=torch.float)
    xcpu = x.cpu()
    xcuda = x.cuda()

    def _subtest(stmt, desc, xcpu=xcpu, xcuda=xcuda):
        t1 = Timer(
            stmt=stmt,
            label='svd',
            sub_label=str(x.size()),
            description=desc,
            globals=dict(globals(), **locals())
            )
        results.append(t1.blocked_autorange())
    
    _subtest('torch.linalg.svd(xcpu,  full_matrices=False)', 'cpu')
    _subtest("torch.linalg.svd(xcuda, full_matrices=False)", 'cuda')

test((100, 10, 10))
test((300, 900, 100))
test((1000, 60, 3))
test((8716, 3, 2))
Compare(results).print()