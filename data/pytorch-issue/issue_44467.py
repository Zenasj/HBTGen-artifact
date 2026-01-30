import torch

py
def t(op_str, # - op name (string)
      subtest_str, # - (sub)test name (string)
      tensor_ctor, # - tensor constructor, takes dtype and device and constructs the tensor to run the op on
      arg_ctor, # - arg constructor, takes dtype and device and constructs op arguments
      half_precision=1e-5, # - torch.half precision
      bfloat16_precision=1e-5, # - torch.bfloat16 precision
      float_precision=1e-5, # - precision to use for all other dtypes
      dtype_list=_types, # - a list of torch dtypes to test the op(s) with
      dtype_cpu_list=[], # - list of torch dtypes to test the op(s) on cpu
      make_inplace_variant=True, # if true the inplace version of the op (op_) is also tested
      decorators=None, # a list of decorators to apply to the test
      self_position=-1,# the position of self in the arg list, -1 means skip function check
      test_out=False, # whether to test the out= version of the operator
):
    return (op_str, subtest_str, tensor_ctor, arg_ctor, half_precision,
            bfloat16_precision, float_precision, dtype_list, dtype_cpu_list,
            make_inplace_variant, decorators, self_position, test_out)

tensor_op_tests = [
    t('add', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1e-2),
    t('add', 'tensor', _small_3d, lambda t, d: [_small_3d(t, d)], 1e-2),
    t('sub', '', _small_3d, lambda t, d: [_number(3.14, 3, t)], 1e-2),
    ...
]
del t