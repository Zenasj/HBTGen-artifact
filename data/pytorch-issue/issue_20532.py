import torch; torch.zeros((), device='cuda')

import tensorflow as tf; tf.constant([])

import cupy; cupy.array([])

import pycuda.driver as cuda; import pycuda.autoinit; cuda.mem_alloc(1)