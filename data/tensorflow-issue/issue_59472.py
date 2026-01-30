from tensorflow.python.framework.ops import EagerTensor
class MyTFTensor(EagerTensor):
    
    @classmethod
    def _from_native(cls, value: tf.Tensor):
        value.__class__ = cls
        return value

import tensorflow as tf

class MyTensor:
    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        return self._tensor

my_tensor = MyTensor(tf.constant(tf.zeros((3, 224, 224))))