import torch

def test_nondeterministic_seeded(func, args, kwargs):
    if func is None or torch.Tag.inplace_view in func.tags:
        return
    results = []
    for i in range(2):
        results.append(func(*args, **kwargs))
    try:
        TestCase.assertEqual(TestCase(), results[0], results[1])
    except AssertionError:
        assert torch.Tag.nondeterministic_seeded in func.tags, f'{func} should be nondeterministic_seeded'
    try:
        TestCase.assertEqual(TestCase(), results[0], results[1], atol=0, rtol=0)
    except AssertionError:
        has_nondeterminism_tag = torch.Tag.nondeterministic_bitwise in func.tags or torch.Tag.nondeterministic_seeded in func.tags
        assert has_nondeterminism_tag, f'{func} should be nondeterministic_bitwise'

...

class TestTagsMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        test_nondeterministic_seeded(func, args, kwargs)

...