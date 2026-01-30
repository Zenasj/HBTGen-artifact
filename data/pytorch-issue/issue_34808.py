import torch.nn as nn

import numpy as np
import tensorflow.compat.v1 as tf

arr = np.array([[0, 1, 2, 3, 4]], dtype='float32')

input = tf.constant(arr)
input4D = tf.reshape(input, [1, 1, 5, 1])
print(tf.image.resize(input4D, [1, 3], method='nearest', align_corners=True)[0,:,:,0])
print(tf.image.resize(input4D, [1, 2], method='nearest', align_corners=True)[0,:,:,0])
# tf.Tensor([[0. 2. 4.]], shape=(1, 3), dtype=float32)
# tf.Tensor([[0. 4.]], shape=(1, 2), dtype=float32)

import torch
import torch.nn.functional as F
arr = torch.from_numpy(arr[None, None, :, :])
print(F.interpolate(arr, (1, 3), mode='nearest')[0,0])
print(F.interpolate(arr, (1, 2), mode='nearest')[0,0])
# tensor([[0., 1., 3.]])
# tensor([[0., 2.]])

import numpy as np
import tensorflow as tf
arr = np.array([[0, 1, 2, 3, 4]], dtype='float32')
input = tf.constant(arr)
input4D = tf.reshape(input, [1, 1, 5, 1])
print(tf.image.resize(input4D, [1, 3], method='nearest')[0,:,:,0])  # [0, 2, 4]
print(tf.image.resize(input4D, [1, 2], method='nearest')[0,:,:,0])  # [1, 3]