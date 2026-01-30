if not TEST_WITH_ROCM:
    inductor_gradient_expected_failures_single_sample["cuda"]["tanh"] = {f16}