DecorateInfo(
    unittest.expectedFailure, "TestOps", "test_python_ref_executor",
    device_type='cuda', active_if=lambda params: params['executor'] == 'nvfuser'
),

# Example: Expect failure if x == 3 and we're in eval mode.
@applyIf(unittest.expectedFailure,
         lambda params: params['x'] == 3 and not params['training'])
@parametrize("x", range(5))
@parametrize("training", [False, True])
def test_foo(self, x, training):
    ...