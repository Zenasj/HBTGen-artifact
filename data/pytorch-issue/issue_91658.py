DecorateInfo(
    unittest.expectedFailure, "TestOps", "test_python_ref_executor",
    device_type='cuda', active_if=lambda params: params['executor'] == 'nvfuser'
),