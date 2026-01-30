import torch

def test_mul_int_oo_nan(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5, do_not_specialize_zero_one=True, positive=None)
        s1 = create_symint(shape_env, 6, do_not_specialize_zero_one=True, positive=None)
        torch._check(s0 >= 0)
        statically_known_true(s0 * s1 > 10)