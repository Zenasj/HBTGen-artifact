import numpy as np

ERROR: test_cond_cpu_complex128 (__main__.TestLinalgCPU)
ERROR: test_cond_cpu_complex64 (__main__.TestLinalgCPU)
ERROR: test_cond_cpu_float32 (__main__.TestLinalgCPU)
ERROR: test_cond_cpu_float64 (__main__.TestLinalgCPU)

input=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])

np.linalg.cond(input,'nuc')

input=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
np.linalg.cond(input,'nuc')